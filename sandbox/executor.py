import optuna
import numpy as np
import ast
import cv2
import time
import inspect
import skimage
import scipy

from tools import global_registry
from core.evaluator import Evaluator
from sandbox.code_checker import AgentCodeChecker, SecurityViolation
from types import SimpleNamespace
from typing import Callable

from RestrictedPython import utility_builtins, safe_builtins

class SandboxExecutor:
    """
    安全、动态执行 LLM 生成的代码。
    """
    _base_namespace = dict(
        np = np,
        cv2 = cv2,
        optuna = optuna,
        skimage = skimage,
        scipy = scipy,
        __builtins__ = dict(
            max = max,
            min = min,
            sum = sum,
            all = all,
            any = any,
            enumerate = enumerate,
            reversed = reversed,
            **safe_builtins
        ),
        cv_wrappers = None,
        vision_metrics = None,
        **utility_builtins
    )

    _func: Callable[[np.ndarray, optuna.Trial], np.ndarray] | None = None
    _code: int | None = None

    _eva_func: Callable[[np.ndarray], float] | None = None
    _eva_code: int | None = None

    def __init__(self, timeout_seconds: int = 5):
        self.timeout = timeout_seconds
        # 占位，防止运行时间过长
        self.registry = global_registry

    @staticmethod
    def get_keys_iters(d):
        stack = [d]
        keys = []
        while stack:
            current = stack.pop()
            for k, v in current.items():
                keys.append(k)
                if isinstance(v, dict):
                    stack.append(v)
        return keys

    @property
    def base_namespace(self):
        return SandboxExecutor.get_keys_iters(self._base_namespace)
    
    def prepare_evaluate_code(self, code_str: str, orig_img: np.ndarray) -> Evaluator:
        exec_context = self._base_namespace.copy()

        exec_context["vision_metrics"] = Evaluator(orig_img)

        # 3. 动态执行
        try:
            # 安全检查
            astree = ast.parse(code_str)
            
            checker = AgentCodeChecker()
            checker.visit(astree)

            exec(code_str, exec_context)

             # 4. 检查是否存在process函数，提取并执行
            process_func: Callable[[np.ndarray, optuna.Trial], np.ndarray] = exec_context.get("evaluate")

            if not process_func:
                raise ValueError("Agent 代码中未定义 evaluate 函数")
            
            sig = inspect.signature(process_func)
            sig.bind(
                np.random.randint(0, 256, (48, 64), dtype=np.uint8),
            )
            
            self._eva_func = process_func
            self._eva_code = hash(code_str)

        except SecurityViolation as e:
            raise
        except TypeError as e:
            raise
        except SyntaxError as e:
            raise
        except Exception as e:
            raise

        return exec_context["vision_metrics"]
    
    def prepare_code(self, code_str: str):
        exec_context = self._base_namespace.copy()
        cv_wrappers_obj = SimpleNamespace()

        for tool_name, tool_data in self.registry._tools.items():
            # 这里 local_namespace["tool_name"] 指向了真实的函数，让动态执行代码可以直接使用 tool_name() 调用
            exec_context[tool_name] = tool_data["func"]
            setattr(cv_wrappers_obj, tool_name, tool_data["func"]) # 绑定到对象属性
    
        exec_context["cv_wrappers"] = cv_wrappers_obj

        # 3. 动态执行
        try:
            # 安全检查
            astree = ast.parse(code_str)
            
            checker = AgentCodeChecker()
            checker.visit(astree)

            exec(code_str, exec_context)

             # 4. 检查是否存在process函数，提取并执行
            process_func: Callable[[np.ndarray, optuna.Trial], np.ndarray] = exec_context.get("process")

            if not process_func:
                raise ValueError("Agent 代码中未定义 process 函数")
            
            sig = inspect.signature(process_func)
            sig.bind(
                np.random.randint(0, 256, (48, 64), dtype=np.uint8),
                optuna.trial.FixedTrial({})
            )
            
            self._func = process_func
            self._code = hash(code_str)

        except SecurityViolation as e:
            raise
        except TypeError as e:
            raise
        except SyntaxError as e:
            raise
        except Exception as e:
            raise

    def test_generated_tools(self, code_str: str, func_name: str, schema: dict) -> Exception | None:
        exec_context = self._base_namespace.copy()
        try:
            # 安全检查
            astree = ast.parse(code_str)
            
            checker = AgentCodeChecker()
            checker.visit(astree)

            exec(code_str, exec_context)

            tool_func: Callable = exec_context.get(func_name)

            if not tool_func:
                raise ValueError(f"Agent 代码中未定义 {func_name} 函数")
            
            test_img = np.random.randint(0, 256, (48, 64), dtype=np.uint8)

            sig = inspect.signature(tool_func)
            sig.bind(test_img)
            
            result = tool_func(test_img)
            if not isinstance(result, np.ndarray):
                raise ValueError("Tool retval is not np.ndarray")
            
            start_time = time.perf_counter_ns()
            for _ in range(5):
                global_registry.tools['Gaussian_Blur']['func'](test_img)
            end_time = time.perf_counter_ns()
            basic_cost = end_time - start_time

            start_time = time.perf_counter_ns()
            for _ in range(5):
                tool_func(test_img)
            end_time = time.perf_counter_ns()
            target_cost = end_time - start_time

            ratio = target_cost / basic_cost
            performance = ""
            if ratio < 2:
                performance = "very fast"
            elif ratio < 10:
                performance = "faster"
            elif ratio < 50:
                performance = "fast"
            elif ratio < 100:
                performance = "medium"
            elif ratio < 300:
                performance = "slow"
            elif ratio < 500:
                performance = "slower"
            elif ratio < 1000:
                performance = "very slow"
            else:
                performance = "slowest"
            
            global_registry.dynamic_register(tool_func, schema, performance)

            return None

        except Exception as e:
            return e

    def execute_pipeline(self, code_str: str, img: np.ndarray, trial: optuna.Trial) -> np.ndarray | Exception:
        """
        使用 exec() 执行 code_str。

        要求 code_str 必须定义了一个 process(img: np.ndarray, trial: optuna.Trial) -> nd.ndarray 函数。

        返回处理后的图像。如果报错，抛出携带详细 Traceback 的异常供 LLM 修复。
        """
        # 创建一个命名空间，注册基础组件
        if not self._func or self._code != hash(code_str):
            self.prepare_code(code_str)

        return self._func(img, trial)
    
    def execute_evaluate(self, code_str: str, img: np.ndarray, orig_img: None | np.ndarray = None) -> float:
        """
        使用 exec() 执行 code_str。

        要求 code_str 必须定义了一个 evaluate(img: np.ndarray) -> nd.ndarray 函数。

        返回处理后的图像。如果报错，抛出携带详细 Traceback 的异常供 LLM 修复。
        """
        # 创建一个命名空间，注册基础组件
        if not self._eva_func or self._eva_code != hash(code_str):
            self.prepare_evaluate_code(code_str, orig_img)

        return self._eva_func(img)
    
    def execute_pipeline_direct(self, code_str: str, img: np.ndarray, params: dict) -> np.ndarray:
        """
        通过注入 Fake Trial 实现参数复用。
        """
        if not self._func or self._code != hash(code_str):
            self.prepare_code(code_str)
            
        # 实例化伪造的 Trial
        fake_trial = optuna.trial.FixedTrial(params)

        return self._func(img, fake_trial)