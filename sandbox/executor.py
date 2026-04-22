import optuna
import numpy as np
import ast
import cv2
import inspect
import skimage

from tools import global_registry
from sandbox.code_checker import AgentCodeChecker, SecurityViolation
from types import SimpleNamespace
from typing import Callable

class SandboxExecutor:
    """
    安全、动态执行 LLM 生成的代码。
    """
    _base_namespace = {
        "np": np,
        "cv2": cv2,
        "optuna": optuna,
        "skimage": skimage,
        "__builtins__": {
            "range": range, 
            "print": print, 
            "float": float, 
            "int": int,
            "str": str
        },
        "cv_wrappers": None
    }

    _func: Callable[[np.ndarray, optuna.Trial], np.ndarray] | None = None
    _code: int | None = None

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
            raise e
        except TypeError as e:
            raise e
        except SyntaxError as e:
            raise e
        except Exception as e:
            raise e

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
    
    def execute_pipeline_direct(self, code_str: str, img: np.ndarray, params: dict) -> np.ndarray:
        """
        通过注入 Fake Trial 实现参数复用。
        """
        if not self._func or self._code != hash(code_str):
            self.prepare_code(code_str)
            
        # 实例化伪造的 Trial
        fake_trial = optuna.trial.FixedTrial(params)

        return self._func(img, fake_trial)