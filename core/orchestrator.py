import numpy as np
import traceback
import optuna

# from agents.planner import PlannerAgent
from sandbox.executor import SandboxExecutor
from agents.coder import CoderAgent
from core.optimizer import BayesianOptimizer
from core.evaluator import Evaluator

from typing import Generator, Callable, Iterable, Literal, overload
from queue import Queue

class Orchestrator:
    """
    管理整个 State 流程与报错重试逻辑 (类似 LangGraph 的大脑)。
    """
    def __init__(self, coder: CoderAgent, max_llm_retries: int = 3):
        # 初始化各个组件
        # self.planner = PlannerAgent(...)
        self.coder = coder
        self.executor = SandboxExecutor()
        self.max_llm_retries = max_llm_retries

    def verify_syntax(self, code_str: str):
        """语法验证"""
        try:
            self.executor.prepare_code(code_str)
        except Exception as e:
            return False, traceback.format_exc()
        return True, None

    def process_stream(self, 
        image: np.ndarray, 
        best_queue: Queue,
        user_prompt: str = '', 
        n_trials: int = 30,
        callbacks: Iterable[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None = None
    ) -> Generator[tuple[
            str, 
            str | tuple[np.ndarray, dict, str]
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

        :param FINISH: 返回最终的结果
        :type FINISH: tuple[np.ndarray, dict, str]
        """
        code_str = ""

        self.evaluator = Evaluator(image)
        yield "INIT.FINISH", None
        self.optimizer = BayesianOptimizer(executor=self.executor, evaluator=self.evaluator)

        user_prompt = f"{user_prompt}\n\n" if user_prompt else "" 
        user_prompt += f"原始图像无参考量化指标:\n{self.evaluator.get_profile_json()}"

        error_log = None
        for attempt in range(self.max_llm_retries):
            yield "CODE.START", None
            for t, chunk in self.coder.execute_stream(user_prompt, [], error_log):
                if t == "FINISH":
                    code_str = chunk
                    yield "CODE.END", None
                elif t == "STREAM.CONTENT":
                    yield "CODE.STREAM", chunk
                elif t == "STREAM.REASONING":
                    yield "CODE.REASONING", chunk

            # 尝试做一次空跑验证语法
            is_syntax_valid, error_log = self.verify_syntax(code_str)
            if is_syntax_valid:
                break
        
        if not is_syntax_valid:
            yield "CODE.ERROR", None
            return

        yield 'OPTUNA.START', None
        optimization_result = self.optimizer.run_inner_loop_stream(
            code_str, image, best_queue, n_trials, callbacks=callbacks
        )
        yield 'OPTUNA.END', None
        
        yield 'FINISH', (optimization_result['best_img'], optimization_result['best_params'], '')
        
    def process(self, 
        image: np.ndarray, 
        user_prompt: str = '', 
        n_trials: int = 30,
        callbacks: Iterable[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None = None
    ) -> tuple[np.ndarray, dict, str]:
        """主执行流程"""
        # 1. 记忆检索 (可选): 查询 ChromDB 看是否有类似任务的成熟代码
        
        # 2. 规划层（可选）：经讨论决定合并到代码层，保留可拓展性
        # plan = self.planner.create_plan(user_prompt)
        
        # 3. 外循环：代码生成与试错
        code_str = ""

        self.evaluator = Evaluator(image)
        self.optimizer = BayesianOptimizer(executor=self.executor, evaluator=self.evaluator)

        user_prompt = f"{user_prompt}\n\n" if user_prompt else "" 
        user_prompt += f"原始图像无参考量化指标:\n{self.evaluator.get_profile_json()}"

        error_log = None
        for attempt in range(self.max_llm_retries):
            code_str = self.coder.execute(user_prompt, [], error_log)
            
            # 尝试做一次空跑验证语法
            is_syntax_valid, error_log = self.verify_syntax(code_str)
            if is_syntax_valid:
                break
        
        if not is_syntax_valid:
            return None, None, "失败: LLM 无法生成正确语法的代码。"

        # 4. 内循环：Optuna 参数寻优
        optimization_result = self.optimizer.run_inner_loop(code_str, image, n_trials, callbacks=callbacks)
        
        # 5. 记忆存储（可选）: 如果得分满意，存入数据库供下次使用
        # if stream:
        #     yield 'FINISH', tuple(optimization_result['best_img'], optimization_result['best_params'], '')
        return optimization_result['best_img'], optimization_result['best_params'], ''