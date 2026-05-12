import numpy as np
import traceback
import optuna
import cv2
import logging

from agents.evaluator import EvaluatorAgent
from agents.coder import CoderAgent
from agents.toolmaker import ToolMakerAgent

from sandbox.executor import SandboxExecutor
from core.optimizer import BayesianOptimizer
from core.evaluator import Evaluator

from typing import Generator, Callable, Iterable, Literal
from queue import Queue

logger = logging.getLogger("Orchestrator")

class Orchestrator:
    """
    管理整个 State 流程与报错重试逻辑 (类似 LangGraph 的大脑)。
    """
    def __init__(self, 
        coder: CoderAgent, 
        evaluator_agent: EvaluatorAgent, 
        toolmaker_agent: ToolMakerAgent,
        max_llm_retries: int = 3,
        process_device: str = "cpu",
        process_profile: str = "balanced",
        device_info: str = ""
    ):
        # 初始化各个组件
        # self.planner = PlannerAgent(...)
        self.coder = coder
        self.executor = SandboxExecutor(
            preferred_device=process_device,
            performance_profile=process_profile,
            device_info=device_info
        )
        self.evaluator_agent = evaluator_agent
        self.toolmaker_agent = toolmaker_agent
        self.max_llm_retries = max_llm_retries
        self.optimizer = None

    def verify_syntax(self, code_str: str):
        """语法验证"""
        try:
            self.executor.prepare_code(code_str)
        except Exception as e:
            return False, traceback.format_exc()
        return True, None

    def toolmaker_stream(
        self,
        tool_request: str,
        search_result: dict | None = None,
        additional_imports: Iterable[str] | None = None,
        additional_packages: Iterable[str] | None = None,
        runtime_context: str | None = None
    ) -> Generator[tuple[
        Literal[
            "CODE_TOOL.START", "CODE_TOOL.END", "CODE_TOOL.STREAM", "CODE_TOOL.TEST",
            "CODE_TOOL.REASONING", "FINISH", "ERROR_RETRY"
        ], 
        str | dict | None
    ], None, None]:
        toolmaker_prompt = f"用户需求: \n{tool_request}"
        if runtime_context:
            toolmaker_prompt += f"\n\n运行时环境约束:\n{runtime_context.strip()}"
        if isinstance(search_result, dict) and 'code_snippets' in search_result and 'summary' in search_result:
            extra_lines = []
            if search_result.get("source"):
                extra_lines.append(f"- 来源: {search_result['source']}")
            if search_result.get("repo_id"):
                extra_lines.append(f"- 仓库/模型ID: {search_result['repo_id']}")
            if search_result.get("dependencies"):
                extra_lines.append(f"- 额外依赖: {search_result['dependencies']}")
            if search_result.get("download_dir"):
                extra_lines.append(f"- 本地下载目录: {search_result['download_dir']}")
            if isinstance(search_result.get("downloaded_files"), list) and search_result["downloaded_files"]:
                extra_lines.append(
                    "- 已下载文件:\n  " + "\n  ".join(str(p) for p in search_result["downloaded_files"])
                )
            if search_result.get("download_error"):
                extra_lines.append(f"- 下载失败信息: {search_result['download_error']}")
            if additional_imports:
                extra_lines.append(f"- 可用额外模块: {', '.join(additional_imports)}")

            search_result_str = f"""
提取到的代码:
```
{search_result['code_snippets']}
```

概述
{search_result['summary']}

附加信息
{chr(10).join(extra_lines) if extra_lines else "无"}
""".strip(' \n')
            toolmaker_prompt += f"\n检索结果: \n{search_result_str}"

        error_log = None
        code_str = ""
        schema = {}

        self.executor.extend_runtime(
            additional_imports=additional_imports,
            additional_packages=additional_packages
        )

        for attempt in range(self.max_llm_retries):
            yield "CODE_TOOL.START", None
            for t, chunk in self.toolmaker_agent.execute_stream(toolmaker_prompt, previous_errors=error_log):
                if t == "FINISH":
                    code_str = chunk['code']
                    schema = chunk['schema']
                    yield "CODE_TOOL.END", None
                elif t == "STREAM.CONTENT":
                    yield "CODE_TOOL.STREAM", chunk
                elif t == "STREAM.REASONING":
                    yield "CODE_TOOL.REASONING", chunk

            try:
                yield "CODE_TOOL.TEST", None
                e = self.executor.test_generated_tools(code_str, schema['name'], schema)
                if e is None:
                    yield "FINISH", {
                        "code": code_str,
                        "schema": schema
                    }
                    break

                toolmaker_prompt = code_str
                error_log = str(e)

                yield "ERROR_RETRY", error_log
            except:
                raise

        yield "ERROR", error_log
       
    def prepare_stream(self,
        image: np.ndarray, 
        model_cache: dict,
        device: str | None = None,
        user_prompt: str = '', 
        max_side: int = 0
    ) -> Generator[tuple[
            Literal[
                'CODE_EVALUATE.START', 'CODE_EVALUATE.END', 'CODE_EVALUATE.STREAM', 
                'CODE_EVALUATE.REASONING', 'FINISH'
            ], 
            str | None
        ], None, None]:
        error_log = None
        code_str = ""

        img_to_eval = None
        if max_side > 0:
            h, w = image.shape[:2]
            if (curr_max_side := max(h, w)) > max_side:
                scale = max_side / curr_max_side
                img_to_eval = cv2.resize(
                    image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
        else:
            img_to_eval = image

        for attempt in range(self.max_llm_retries):
            yield "CODE_EVALUATE.START", None
            for t, chunk in self.evaluator_agent.execute_stream(user_prompt):
                if t == "FINISH":
                    code_str = chunk
                    yield "CODE_EVALUATE.END", None
                elif t == "STREAM.CONTENT":
                    yield "CODE_EVALUATE.STREAM", chunk
                elif t == "STREAM.REASONING":
                    yield "CODE_EVALUATE.REASONING", chunk

            # 尝试做一次空跑验证语法
            try:
                self.evaluator = Evaluator(img_to_eval, model_cache, device)
                self.executor.prepare_evaluate_code(code_str, self.evaluator)
                self.executor.execute_evaluate(code_str, image, self.evaluator)
                break
            except:
                raise
       
        yield "FINISH", code_str

    def process_stream(self, 
        image: np.ndarray, 
        evaluate_code_str: str,
        best_queue: Queue,
        user_prompt: str = '', 
        n_trials: int = 30,
        callbacks: Iterable[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None = None,
        error_log: str = '',
        max_side: int = 0
    ) -> Generator[
        tuple[
            Literal[
                "INIT.FINISH", 
                "CODE.START", "CODE.REASONING", "CODE.END", "CODE.ERROR", 
                'OPTUNA.START', 'OPTUNA.END'
            ], 
            str | None
        ] | tuple[
            Literal['FINISH'],
            # ===== [修改] 返回值增加n_trials_used字段 =====
            tuple[np.ndarray, dict, str, int]
        ] | tuple[
            Literal['TOOL_REQUEST'],
            dict
        ], None, None]:
        """
        流式主处理流程

        以信令形式迭代，本函数发出的信令有

        :param CODE.STREAM: CoderAgent的流式返回
        :type CODE.STREAM: str

        :param CODE.REASONING: CoderAgent的思考内容的流式返回（如果有）
        :type CODE.REASONING: str

        :param CODE.END: CoderAgent返回完成
        :type CODE.END: None

        :param CODE.ERROR: CoderAgent返回的代码无法正常运行
        :type CODE.ERROR: None

        :param OPTUNA.START: Optuna开始调优
        :type OPTUNA.START: None

        :param OPTUNA.END: Optuna调优结束
        :type OPTUNA.END: None

        :param FINISH: 返回最终的结果 (best_img, best_params, log, n_trials_used)
        :type FINISH: tuple[np.ndarray, dict, str, int]
        """

        if not hasattr(self, "evaluator"):
            raise RuntimeError("尚未初始化，必须先调用 prepare_stream")

        code_str = ""

        user_prompt = f"{user_prompt}\n\n" if user_prompt else "" 
        
        self.optimizer = BayesianOptimizer(executor=self.executor)

        img_to_opti = None
        if max_side > 0:
            h, w = image.shape[:2]
            if (curr_max_side := max(h, w)) > max_side:
                scale = max_side / curr_max_side
                img_to_opti = cv2.resize(
                    image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
        else:
            img_to_opti = image

        logger.info(f"调优图像大小: {img_to_opti.shape[1]}x{img_to_opti.shape[0]}, 通道数: {img_to_opti.shape[2]}")

        user_prompt += f"调优图像大小: {img_to_opti.shape[1]}x{img_to_opti.shape[0]}, 通道数: {img_to_opti.shape[2]}"

        yield "INIT.FINISH", None

        for attempt in range(self.max_llm_retries):
            yield "CODE.START", None
            for t, chunk in self.coder.execute_stream(user_prompt, "", error_log):
                if t == "FINISH":
                    code_str = chunk
                    if isinstance(code_str, str):
                        yield "CODE.END", chunk
                elif t == "STREAM.CONTENT":
                    yield "CODE.STREAM", chunk
                elif t == "STREAM.REASONING":
                    yield "CODE.REASONING", chunk

            if isinstance(code_str, dict):
                yield "TOOL_REQUEST", code_str
                return

            # 尝试做一次空跑验证语法
            is_syntax_valid, error_log = self.verify_syntax(code_str)
            if is_syntax_valid:
                break
        
        if not is_syntax_valid:
            yield "CODE.ERROR", None
            return

        yield 'OPTUNA.START', None
        optimization_result = self.optimizer.run_inner_loop_stream(
            code_str, evaluate_code_str, img_to_opti, image,
            best_queue, n_trials, callbacks=callbacks, 
        )
        yield 'OPTUNA.END', None
        
        # ===== [新增] 提取实际使用的trial数并返回 =====
        n_trials_used = optimization_result.get('n_trials_used', n_trials)
        yield 'FINISH', (optimization_result['best_img'], optimization_result['best_params'], '', n_trials_used)
