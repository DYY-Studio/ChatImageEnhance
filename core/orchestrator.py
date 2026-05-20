import numpy as np
import traceback
import optuna
import cv2
import logging
import re

from agents.evaluator import EvaluatorAgent
from agents.coder import CoderAgent
from agents.toolmaker import ToolMakerAgent

from sandbox.executor import SandboxExecutor
from core.optimizer import BayesianOptimizer
from core.evaluator import Evaluator

from typing import Generator, Callable, Iterable, Literal
from collections import deque

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
        allow_learning_process: bool = True,
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
        self.allow_learning_process = bool(allow_learning_process)
        self.optimizer = None

    _UNFIXABLE_ERROR_MARKERS = (
        "LLM 调用最终失败",
        "LLM 调用失败",
        "程序错误:",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "RateLimitError",
        "pip dynamic install failed",
        "Unsupported dependency source",
        "Unsafe dependency option",
        "Invalid dependency spec",
        "Empty dependency spec",
        "Permission denied",
        "No space left",
    )

    @staticmethod
    def _exception_details(exc: Exception) -> str:
        return traceback.format_exc()

    @staticmethod
    def _compact_error(error: str | None, limit: int = 1200) -> str:
        text = str(error or "").strip()
        if len(text) <= limit:
            return text
        return text[-limit:]

    @classmethod
    def _is_auto_fixable_error(cls, error: Exception | str | None) -> bool:
        text = str(error or "")
        if not text.strip():
            return False
        return not any(marker in text for marker in cls._UNFIXABLE_ERROR_MARKERS)

    @classmethod
    def _fatal_error_message(cls, stage: str, error: str | None) -> str:
        detail = cls._compact_error(error)
        return (
            f"{stage}失败，且该错误无法通过重写 LLM 生成代码自动修复，已停止运行。\n\n"
            f"错误详情：\n{detail or '无详细错误信息'}"
        )

    @classmethod
    def _retry_error_message(cls, stage: str, error: str | None) -> str:
        detail = cls._compact_error(error)
        return f"{stage}失败，已把错误反馈给 LLM 重新生成。\n\n{detail}"

    @staticmethod
    def _schema_with_model_assets(schema: dict, search_result: dict | None) -> dict:
        if not isinstance(schema, dict):
            return schema
        if not isinstance(search_result, dict):
            return schema

        enriched_schema = dict(schema)
        source = str(search_result.get("source") or "").strip().lower()
        if source in {"github", "huggingface", "modelscope"}:
            enriched_schema["source"] = source

        repo_id = str(search_result.get("repo_id") or "").strip()
        if repo_id:
            enriched_schema["repo_id"] = repo_id

        asset_urls = search_result.get("asset_urls")
        if isinstance(asset_urls, list) and asset_urls:
            schema_asset_urls = []
            for item in asset_urls:
                url = str(item.get("url") if isinstance(item, dict) else item).strip()
                if url:
                    schema_asset_urls.append(url)
            if schema_asset_urls:
                enriched_schema["asset_urls"] = schema_asset_urls
        asset_files = search_result.get("asset_files")
        if isinstance(asset_files, list) and asset_files:
            enriched_schema["asset_files"] = [str(item) for item in asset_files if str(item).strip()]

        return enriched_schema

    @staticmethod
    def _filter_learning_dependencies(
        imports: Iterable[str] | None,
        packages: Iterable[str] | None
    ) -> tuple[list[str] | None, list[str] | None]:
        blocked_import_roots = {"torch", "torchvision", "transformers", "diffusers", "modelscope", "huggingface_hub"}
        blocked_package_roots = {
            "torch", "torchvision", "transformers", "diffusers", "modelscope", "huggingface-hub", "huggingface_hub"
        }

        filtered_imports = None
        if imports is not None:
            filtered_imports = [
                imp for imp in imports
                if str(imp).strip()
                and str(imp).strip().split(".", maxsplit=1)[0].lower() not in blocked_import_roots
            ]

        filtered_packages = None
        if packages is not None:
            def _pkg_root(value: str) -> str:
                token = str(value).strip().split("[", maxsplit=1)[0].strip()
                token = re.split(r"(==|!=|>=|<=|>|<|~=)", token, maxsplit=1)[0].strip()
                return token.lower()

            filtered_packages = [
                pkg for pkg in packages
                if str(pkg).strip()
                and _pkg_root(str(pkg)) not in blocked_package_roots
            ]

        return filtered_imports, filtered_packages

    def verify_syntax(self, code_str: str):
        """语法验证"""
        try:
            self.executor.prepare_code(code_str)
        except Exception as e:
            return False, traceback.format_exc()
        return True, None

    @staticmethod
    def _validate_tool_request(request: dict) -> tuple[bool, str | None]:
        if not isinstance(request, dict):
            return False, "工具请求必须是 JSON 对象"
        if request.get("status") != "NEED_NEW_TOOL":
            return False, "工具请求 status 必须为 NEED_NEW_TOOL"
        description = str(request.get("description") or "").strip()
        if not description:
            return False, "工具请求缺少 description"
        tool_name = str(request.get("tool_name") or "").strip()
        if tool_name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tool_name):
            return False, "工具请求 tool_name 必须是合法 Python 标识符"
        return True, None

    @staticmethod
    def _sanitize_toolmaker_runtime_context(runtime_context: str | None) -> str:
        if not runtime_context:
            return ""
        return "\n".join(
            line
            for line in str(runtime_context).splitlines()
            if not line.strip().startswith(("处理算子偏好:", "处理算子偏好："))
        ).strip()

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
            "CODE_TOOL.REASONING", "FINISH", "ERROR_RETRY", "FATAL_ERROR"
        ], 
        str | dict | None
    ], None, None]:
        toolmaker_prompt = f"用户需求: \n{tool_request}"
        if runtime_context:
            toolmaker_runtime_context = self._sanitize_toolmaker_runtime_context(runtime_context)
            if toolmaker_runtime_context:
                toolmaker_prompt += f"\n\n运行时环境约束:\n{toolmaker_runtime_context}"
        if isinstance(search_result, dict) and 'code_snippets' in search_result and 'summary' in search_result:
            extra_lines = []
            if search_result.get("source"):
                extra_lines.append(f"- 来源: {search_result['source']}")
            if search_result.get("repo_id"):
                extra_lines.append(f"- 仓库/模型ID: {search_result['repo_id']}")
            if search_result.get("dependencies"):
                extra_lines.append(f"- 额外依赖: {search_result['dependencies']}")
            if search_result.get("download_dir"):
                extra_lines.append(
                    "- 模型缓存: 已按 source/repo_id 预下载；repo-id 接口可直接使用缓存，"
                    "本地文件路径加载场景才需要 `model_dir`"
                )
            if isinstance(search_result.get("require_files"), list) and search_result["require_files"]:
                extra_lines.append(
                    "- 需要的仓库内文件:\n  " + "\n  ".join(str(p) for p in search_result["require_files"])
                )
            if isinstance(search_result.get("asset_files"), list) and search_result["asset_files"]:
                extra_lines.append(
                    "- 已下载的外部资产文件（位于运行时注入的 model_dir 下）:\n  "
                    + "\n  ".join(str(p) for p in search_result["asset_files"])
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

        runtime_imports = additional_imports
        runtime_packages = additional_packages
        if not self.allow_learning_process:
            runtime_imports, runtime_packages = self._filter_learning_dependencies(
                additional_imports, additional_packages
            )

        if hasattr(self.toolmaker_agent, "set_additional_imports"):
            self.toolmaker_agent.set_additional_imports(runtime_imports)

        if isinstance(search_result, dict) and search_result.get("download_error"):
            yield "FATAL_ERROR", self._fatal_error_message(
                "模型资产准备",
                str(search_result.get("download_error") or "模型资产下载失败")
            )
            return

        try:
            self.executor.extend_runtime(
                additional_imports=runtime_imports,
                additional_packages=runtime_packages
            )
        except Exception as e:
            details = self._exception_details(e)
            yield "FATAL_ERROR", self._fatal_error_message("运行时依赖准备", details)
            return

        for attempt in range(self.max_llm_retries):
            yield "CODE_TOOL.START", None
            try:
                for t, chunk in self.toolmaker_agent.execute_stream(toolmaker_prompt, previous_errors=error_log):
                    if t == "FINISH":
                        if not isinstance(chunk, dict):
                            raise ValueError("ToolMakerAgent 未返回 JSON 对象")
                        if not isinstance(chunk.get("schema"), dict) or not str(chunk.get("code") or "").strip():
                            raise ValueError("ToolMakerAgent 返回缺少 code 或 schema")
                        code_str = chunk['code']
                        schema = self._schema_with_model_assets(chunk['schema'], search_result)
                        if not isinstance(schema.get("parameters"), dict):
                            raise ValueError("ToolMakerAgent schema.parameters 必须是对象")
                        yield "CODE_TOOL.END", None
                    elif t == "STREAM.CONTENT":
                        yield "CODE_TOOL.STREAM", chunk
                    elif t == "STREAM.REASONING":
                        yield "CODE_TOOL.REASONING", chunk
            except Exception as e:
                details = self._exception_details(e)
                if not self._is_auto_fixable_error(details):
                    yield "FATAL_ERROR", self._fatal_error_message("工具代码生成", details)
                    return
                error_log = details
                yield "ERROR_RETRY", self._retry_error_message("工具代码生成", error_log)
                continue

            try:
                yield "CODE_TOOL.TEST", None
                e = self.executor.test_generated_tools(code_str, schema['name'], schema)
                if e is None:
                    yield "FINISH", {
                        "code": code_str,
                        "schema": schema
                    }
                    return

                toolmaker_prompt = code_str
                error_log = self._compact_error(str(e), 4000)

                yield "ERROR_RETRY", self._retry_error_message("工具代码测试", error_log)
            except Exception as e:
                details = self._exception_details(e)
                if not self._is_auto_fixable_error(details):
                    yield "FATAL_ERROR", self._fatal_error_message("工具代码测试", details)
                    return
                error_log = details
                yield "ERROR_RETRY", self._retry_error_message("工具代码测试", error_log)

        yield "FATAL_ERROR", self._fatal_error_message("工具代码生成", error_log or "已达到最大自动修复次数")
       
    def prepare_stream(self,
        image: np.ndarray, 
        model_cache: dict,
        device: str | None = None,
        user_prompt: str = '', 
        max_side: int = 0
    ) -> Generator[tuple[
            Literal[
                'CODE_EVALUATE.START', 'CODE_EVALUATE.END', 'CODE_EVALUATE.STREAM', 
                'CODE_EVALUATE.REASONING', 'CODE_EVALUATE.ERROR_RETRY', 'FINISH', 'FATAL_ERROR'
            ], 
            str | None
        ], None, None]:
        error_log = None
        code_str = ""

        img_to_eval = image
        if max_side > 0:
            h, w = image.shape[:2]
            if (curr_max_side := max(h, w)) > max_side:
                scale = max_side / curr_max_side
                img_to_eval = cv2.resize(
                    image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )

        for attempt in range(self.max_llm_retries):
            yield "CODE_EVALUATE.START", None
            try:
                for t, chunk in self.evaluator_agent.execute_stream(user_prompt, previous_errors=error_log):
                    if t == "FINISH":
                        code_str = chunk
                        yield "CODE_EVALUATE.END", None
                    elif t == "STREAM.CONTENT":
                        yield "CODE_EVALUATE.STREAM", chunk
                    elif t == "STREAM.REASONING":
                        yield "CODE_EVALUATE.REASONING", chunk
            except Exception as e:
                details = self._exception_details(e)
                if not self._is_auto_fixable_error(details):
                    yield "FATAL_ERROR", self._fatal_error_message("评价代码生成", details)
                    return
                error_log = details
                yield "CODE_EVALUATE.ERROR_RETRY", self._retry_error_message("评价代码生成", error_log)
                continue

            # 尝试做一次空跑验证语法
            try:
                # 释放旧 Evaluator 的 GPU 资源（如重试时）
                if hasattr(self, 'evaluator') and self.evaluator is not None:
                    if hasattr(self.evaluator, 'cleanup'):
                        try:
                            self.evaluator.cleanup()
                        except Exception:
                            pass
                self.evaluator = Evaluator(img_to_eval, model_cache, device)
                self.executor.prepare_evaluate_code(code_str, self.evaluator)
                self.executor.execute_evaluate(code_str, img_to_eval, self.evaluator)
                yield "FINISH", code_str
                return
            except Exception as e:
                details = self._exception_details(e)
                if not self._is_auto_fixable_error(details):
                    yield "FATAL_ERROR", self._fatal_error_message("评价代码验证", details)
                    return
                error_log = details
                yield "CODE_EVALUATE.ERROR_RETRY", self._retry_error_message("评价代码验证", error_log)
       
        yield "FATAL_ERROR", self._fatal_error_message("评价代码生成", error_log or "已达到最大自动修复次数")

    def process_stream(self, 
        image: np.ndarray, 
        evaluate_code_str: str,
        best_queue: deque[np.ndarray],
        user_prompt: str = '', 
        n_trials: int = 30,
        callbacks: Iterable[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None = None,
        error_log: str = '',
        max_side: int = 0
    ) -> Generator[
        tuple[
            Literal[
                "INIT.FINISH", 
                "CODE.START", "CODE.STREAM", "CODE.REASONING", "CODE.END", "CODE.ERROR", 
                'OPTUNA.START', 'OPTUNA.END', 'FATAL_ERROR'
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

        img_to_opti = image
        if max_side > 0:
            h, w = image.shape[:2]
            if (curr_max_side := max(h, w)) > max_side:
                scale = max_side / curr_max_side
                img_to_opti = cv2.resize(
                    image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )

        logger.info(f"调优图像大小: {img_to_opti.shape[1]}x{img_to_opti.shape[0]}, 通道数: {img_to_opti.shape[2]}")

        user_prompt += f"调优图像大小: {img_to_opti.shape[1]}x{img_to_opti.shape[0]}, 通道数: {img_to_opti.shape[2]}"

        yield "INIT.FINISH", None

        for attempt in range(self.max_llm_retries):
            code_str = ""
            yield "CODE.START", None
            try:
                for t, chunk in self.coder.execute_stream(user_prompt, "", error_log):
                    if t == "FINISH":
                        code_str = chunk
                        if isinstance(code_str, str):
                            yield "CODE.END", chunk
                    elif t == "STREAM.CONTENT":
                        yield "CODE.STREAM", chunk
                    elif t == "STREAM.REASONING":
                        yield "CODE.REASONING", chunk
            except Exception as e:
                details = self._exception_details(e)
                if not self._is_auto_fixable_error(details):
                    yield "FATAL_ERROR", self._fatal_error_message("处理代码生成", details)
                    return
                error_log = details
                yield "CODE.ERROR", self._retry_error_message("处理代码生成", error_log)
                continue

            if isinstance(code_str, dict):
                valid_request, request_error = self._validate_tool_request(code_str)
                if not valid_request:
                    error_log = request_error or "工具请求格式不合法"
                    yield "CODE.ERROR", self._retry_error_message("工具请求验证", error_log)
                    continue
                yield "TOOL_REQUEST", code_str
                return

            # 尝试做一次空跑验证语法
            is_syntax_valid, error_log = self.verify_syntax(code_str)
            if not is_syntax_valid:
                yield "CODE.ERROR", self._retry_error_message("处理代码验证", error_log)
                continue

            self.optimizer = BayesianOptimizer(executor=self.executor)
            yield 'OPTUNA.START', None
            try:
                optimization_result = self.optimizer.run_inner_loop_stream(
                    code_str, evaluate_code_str, img_to_opti, image,
                    best_queue, n_trials, callbacks=callbacks, 
                )
            except Exception as e:
                yield 'OPTUNA.END', None
                details = self._exception_details(e)
                if not self._is_auto_fixable_error(details):
                    yield "FATAL_ERROR", self._fatal_error_message("图像处理执行", details)
                    return
                error_log = details
                yield "CODE.ERROR", self._retry_error_message("图像处理执行", error_log)
                continue
            yield 'OPTUNA.END', None

            if optimization_result.get("error") or optimization_result.get("best_img") is None:
                error_log = optimization_result.get("error") or "未得到可用增强结果。"
                if not self._is_auto_fixable_error(error_log):
                    yield "FATAL_ERROR", self._fatal_error_message("图像处理执行", error_log)
                    return
                yield "CODE.ERROR", self._retry_error_message("图像处理执行", error_log)
                continue
            
            # ===== [新增] 提取实际使用的trial数并返回 =====
            n_trials_used = optimization_result.get('n_trials_used', n_trials)
            yield 'FINISH', (optimization_result['best_img'], optimization_result['best_params'], '', n_trials_used)
            return

        yield "FATAL_ERROR", self._fatal_error_message("处理代码生成", error_log or "已达到最大自动修复次数")
