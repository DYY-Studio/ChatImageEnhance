import numpy as np
import traceback
import optuna

from agents.evaluator import EvaluatorAgent
from sandbox.executor import SandboxExecutor
from agents.coder import CoderAgent
from core.optimizer import BayesianOptimizer
# from core.evaluator import Evaluator

from typing import Generator, Callable, Iterable, Literal, overload
from queue import Queue

class Orchestrator:
    """
    管理整个 State 流程与报错重试逻辑 (类似 LangGraph 的大脑)。
    """
    def __init__(self, coder: CoderAgent, evaluator_agent: EvaluatorAgent, max_llm_retries: int = 3):
        # 初始化各个组件
        # self.planner = PlannerAgent(...)
        self.coder = coder
        self.executor = SandboxExecutor()
        self.evaluator_agent = evaluator_agent
        self.max_llm_retries = max_llm_retries

    def verify_syntax(self, code_str: str):
        """语法验证"""
        try:
            self.executor.prepare_code(code_str)
        except Exception as e:
            return False, traceback.format_exc()
        return True, None
    
    def prepare_stream(self,
        image: np.ndarray, 
        user_prompt: str = '', 
    ) -> Generator[tuple[
            str, str
        ], None, None]:
        error_log = None
        code_str = ""

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
                self.evaluator = self.executor.prepare_evaluate_code(code_str, image)
                self.executor.execute_evaluate(code_str, image, image)
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
        error_log: str = ''
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

        if not hasattr(self, "evaluator"):
            raise RuntimeError("尚未初始化，必须先调用 prepare_stream")

        code_str = ""

        user_prompt = f"{user_prompt}\n\n" if user_prompt else "" 
        
        self.optimizer = BayesianOptimizer(executor=self.executor)
        yield "INIT.FINISH", None

        for attempt in range(self.max_llm_retries):
            yield "CODE.START", None
            for t, chunk in self.coder.execute_stream(user_prompt, "", error_log):
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
            code_str, evaluate_code_str, image, best_queue, n_trials, callbacks=callbacks
        )
        yield 'OPTUNA.END', None
        
        yield 'FINISH', (optimization_result['best_img'], optimization_result['best_params'], '')
    