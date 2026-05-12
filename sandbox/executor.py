import optuna
import numpy as np
import ast
import time
import inspect
import builtins as py_builtins

from tools import global_registry
from core.evaluator import Evaluator
from sandbox.code_checker import (
    AgentCodeChecker,
    DEFAULT_ALLOWED_IMPORT_PREFIXES,
    SecurityViolation,
    is_allowed_import_path,
)
from sandbox.runtime_dependencies import RuntimeDependencyManager
from types import ModuleType, SimpleNamespace
from typing import Callable, Iterable

from RestrictedPython import utility_builtins, safe_builtins

class _ReadOnlyBuiltins(dict):
    def _deny(self, *args, **kwargs):
        raise TypeError("builtins is read-only")

    __setitem__ = _deny
    __delitem__ = _deny
    clear = _deny
    pop = _deny
    popitem = _deny
    setdefault = _deny
    update = _deny

class SandboxExecutor:
    """
    安全、动态执行 LLM 生成的代码。
    """
    _MEMORY_PARAM_KEYWORDS = (
        "tile", "patch", "chunk", "batch", "window", "crop", "sw_batch", "micro_batch"
    )
    _PROFILE_SCALE = {
        "fast": 1.0,
        "balanced": 0.9,
        "low_memory": 0.7
    }

    _func: Callable[[np.ndarray, optuna.Trial], np.ndarray] | None = None
    _code: int | None = None
    _func_accepts_cache: bool | None = None

    _eva_func: Callable[[np.ndarray], float] | None = None
    _eva_code: int | None = None

    def __init__(self, 
        timeout_seconds: int = 5, 
        additional_imports: Iterable[str] | None = None,
        additional_packages: Iterable[str] | None = None,
        preferred_device: str = "cpu",
        performance_profile: str = "balanced",
        device_info: str = ""
    ):
        self.timeout = timeout_seconds
        # 占位，防止运行时间过长
        self.registry = global_registry
        self.preferred_device = preferred_device
        self.performance_profile = performance_profile if performance_profile in self._PROFILE_SCALE else "balanced"
        self.device_info = device_info
        self._base_namespace = self._create_base_namespace()
        self._runtime_deps = RuntimeDependencyManager(self._base_namespace)
        builtins_dict = dict(self._base_namespace.get("__builtins__", {}))
        builtins_dict["__import__"] = self._safe_import
        self._base_namespace["__builtins__"] = _ReadOnlyBuiltins(builtins_dict)
        self.extend_runtime(additional_imports=additional_imports, additional_packages=additional_packages)

    @staticmethod
    def _create_base_namespace() -> dict:
        return dict(
            np=np,
            optuna=optuna,
            __builtins__=dict(
                max=max,
                min=min,
                sum=sum,
                all=all,
                any=any,
                enumerate=enumerate,
                reversed=reversed,
                list=list, # 危险可变对象，考虑进行安全封装，但目前先维持这样
                dict=dict, # 危险可变对象，考虑进行安全封装，但目前先维持这样
                **safe_builtins
            ),
            cv_wrappers=None,
            vision_metrics=None,
            **utility_builtins
        )

    def _load_core_optional_modules(self):
        self._runtime_deps.load_core_optional_modules()

    def _build_allowed_import_prefixes(self) -> tuple[str, ...]:
        prefixes = set(DEFAULT_ALLOWED_IMPORT_PREFIXES)
        prefixes.update(self._runtime_deps.get_allowed_import_prefixes())
        for value in self._base_namespace.values():
            if isinstance(value, ModuleType):
                mod_name = str(getattr(value, "__name__", "") or "").strip()
                if not mod_name:
                    continue
                prefixes.add(mod_name)
                if "." in mod_name:
                    prefixes.add(mod_name.split(".", maxsplit=1)[0])
        return tuple(sorted(prefixes))

    def _safe_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level and level > 0:
            raise SecurityViolation("禁止使用相对导入")

        mod_name = str(name or "").strip()
        allowed_prefixes = self._build_allowed_import_prefixes()
        if not is_allowed_import_path(mod_name, allowed_prefixes):
            raise SecurityViolation(f"禁止导入非白名单模块: {mod_name}")

        for item in (fromlist or ()):
            token = str(item or "").strip()
            if token == "*" or token.startswith("__"):
                raise SecurityViolation(f"禁止导入符号: {token}")

        return py_builtins.__import__(mod_name, globals, locals, fromlist, level)

    def set_runtime_preferences(
        self,
        preferred_device: str | None = None,
        performance_profile: str | None = None,
        device_info: str | None = None
    ):
        if preferred_device:
            self.preferred_device = preferred_device
        if performance_profile:
            self.performance_profile = (
                performance_profile if performance_profile in self._PROFILE_SCALE else "balanced"
            )
        if device_info is not None:
            self.device_info = device_info

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            key in msg for key in (
                "out of memory",
                "cuda error: out of memory",
                "mps backend out of memory",
                "hip out of memory",
                "xpu out of memory",
                "cublas_status_alloc_failed",
            )
        )

    @staticmethod
    def _is_device_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            key in msg for key in (
                "cuda is not available",
                "torch not compiled with cuda",
                "invalid device",
                "device type",
                "mps backend is not available",
                "xpu is not available",
                "npu is not available",
            )
        )

    @staticmethod
    def _release_device_memory():
        RuntimeDependencyManager.release_device_memory()

    def _ensure_runtime_meta(self, runtime_cache: dict):
        runtime_meta = runtime_cache.setdefault("__runtime__", {})
        runtime_meta["preferred_device"] = self.preferred_device
        runtime_meta["performance_profile"] = self.performance_profile
        runtime_meta["device_info"] = self.device_info
        runtime_cache.setdefault("__runtime_oom_level__", 0)
        runtime_cache.setdefault("__runtime_param_lock__", {})

    def _next_oom_level(self, runtime_cache: dict, max_level: int = 5) -> bool:
        curr = int(runtime_cache.get("__runtime_oom_level__", 0))
        if curr >= max_level:
            return False
        runtime_cache["__runtime_oom_level__"] = curr + 1
        return True

    def _is_memory_param(self, name: str) -> bool:
        lname = name.lower()
        return any(keyword in lname for keyword in self._MEMORY_PARAM_KEYWORDS)

    def _get_profile_factor(self) -> float:
        return self._PROFILE_SCALE.get(self.performance_profile, 0.9)

    def _get_adaptive_factor(self, runtime_cache: dict, is_memory_param: bool) -> float:
        if not is_memory_param:
            return 1.0
        profile_factor = self._get_profile_factor()
        oom_level = int(runtime_cache.get("__runtime_oom_level__", 0))
        oom_factor = 0.75 ** oom_level
        return max(0.15, profile_factor * oom_factor)

    @staticmethod
    def _align_int_step(value: int, low: int, high: int, step: int) -> int:
        if step <= 1:
            return int(min(max(value, low), high))
        aligned = low + ((int(value) - low) // step) * step
        if aligned < low:
            aligned = low
        if aligned > high:
            aligned = high
        return int(aligned)

    class _AdaptiveTrial:
        def __init__(self, outer: "SandboxExecutor", base_trial: optuna.Trial, runtime_cache: dict):
            self._outer = outer
            self._base = base_trial
            self._cache = runtime_cache
            self._locks = runtime_cache.setdefault("__runtime_param_lock__", {})
            self._pending_locks: dict[str, int | float | str | bool] = {}

        @property
        def pending_locks(self):
            return self._pending_locks

        def _record_lock(self, name: str, value):
            if self._outer._is_memory_param(name):
                self._pending_locks[name] = value

        def _apply_lock_or_adjust(self, name: str, value, low=None, high=None, step=None):
            if name in self._locks:
                return self._locks[name]

            is_memory = self._outer._is_memory_param(name)
            factor = self._outer._get_adaptive_factor(self._cache, is_memory)
            if factor >= 0.999:
                self._record_lock(name, value)
                return value

            adjusted = value
            if isinstance(value, (int, np.integer)):
                adjusted = int(round(float(value) * factor))
                if low is not None and high is not None:
                    adjusted = self._outer._align_int_step(adjusted, int(low), int(high), int(step or 1))
            elif isinstance(value, (float, np.floating)):
                adjusted = float(value) * factor
                if low is not None:
                    adjusted = max(float(low), adjusted)
                if high is not None:
                    adjusted = min(float(high), adjusted)
                if step:
                    try:
                        step_f = float(step)
                        if step_f > 0:
                            adjusted = float(low) + round((adjusted - float(low)) / step_f) * step_f
                    except Exception:
                        pass

            self._record_lock(name, adjusted)
            return adjusted

        def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False):
            value = self._base.suggest_int(name, low, high, step=step, log=log)
            return self._apply_lock_or_adjust(name, value, low=low, high=high, step=step)

        def suggest_float(
            self,
            name: str,
            low: float,
            high: float,
            step: float | None = None,
            log: bool = False
        ):
            value = self._base.suggest_float(name, low, high, step=step, log=log)
            return self._apply_lock_or_adjust(name, value, low=low, high=high, step=step)

        def suggest_categorical(self, name: str, choices):
            if name in self._locks:
                return self._locks[name]
            value = self._base.suggest_categorical(name, choices)
            if self._outer._is_memory_param(name):
                try:
                    numeric_choices = [c for c in choices if isinstance(c, (int, float, np.integer, np.floating))]
                    if numeric_choices:
                        factor = self._outer._get_adaptive_factor(self._cache, True)
                        target = float(value) * factor if isinstance(value, (int, float, np.integer, np.floating)) else None
                        if target is not None:
                            candidate = min(numeric_choices, key=lambda x: abs(float(x) - target))
                            value = candidate
                except Exception:
                    pass
            self._record_lock(name, value)
            return value

        def __getattr__(self, item):
            return getattr(self._base, item)

    @staticmethod
    def _canonical_dist_name(name: str) -> str:
        return RuntimeDependencyManager._canonical_dist_name(name)

    @staticmethod
    def _is_identifier(name: str) -> bool:
        return RuntimeDependencyManager._is_identifier(name)

    @classmethod
    def _extract_dist_name(cls, requirement: str) -> str:
        return RuntimeDependencyManager._extract_dist_name(requirement)

    @classmethod
    def _normalize_package_name(cls, package: str) -> str:
        return RuntimeDependencyManager._normalize_package_name(package)

    @classmethod
    def _resolve_import_from_installed_dist(cls, dist_name: str) -> str | None:
        return RuntimeDependencyManager._resolve_import_from_installed_dist(dist_name)

    @classmethod
    def _resolve_import_name(cls, name: str) -> str | None:
        return RuntimeDependencyManager._resolve_import_name(name)

    @classmethod
    def _get_installed_version(cls, dist_name: str, refresh: bool = False) -> str | None:
        return RuntimeDependencyManager._get_installed_version(dist_name, refresh=refresh)

    @classmethod
    def _get_distribution_candidates(cls, dist_name: str) -> list[str]:
        return RuntimeDependencyManager._get_distribution_candidates(dist_name)

    @classmethod
    def _requirement_satisfied(cls, requirement: str) -> tuple[bool, str]:
        return RuntimeDependencyManager._requirement_satisfied(requirement)

    @classmethod
    def _candidate_import_names_for_dist(cls, dist_name: str) -> set[str]:
        return RuntimeDependencyManager._candidate_import_names_for_dist(dist_name)

    def _best_effort_release_for_install(self, dist_name: str):
        self._runtime_deps._best_effort_release_for_install(dist_name)

    def _install_packages(self, packages: Iterable[str] | None):
        self._runtime_deps.install_packages(packages)

    def _import_modules(self, imports: Iterable[str] | None):
        self._runtime_deps.import_modules(imports)

    def _recover_missing_module_dependency(self, exc: Exception) -> bool:
        return self._runtime_deps.recover_missing_module_dependency(exc)

    def extend_runtime(
        self,
        additional_imports: Iterable[str] | None = None,
        additional_packages: Iterable[str] | None = None
    ):
        self._runtime_deps.extend_runtime(
            additional_imports=additional_imports,
            additional_packages=additional_packages
        )

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
    
    def prepare_evaluate_code(self, code_str: str, evaluator: Evaluator):
        for _ in range(2):
            exec_context = self._base_namespace.copy()
            exec_context["vision_metrics"] = evaluator

            try:
                astree = ast.parse(code_str)
                checker = AgentCodeChecker(self._build_allowed_import_prefixes())
                checker.visit(astree)

                exec(code_str, exec_context)

                process_func: Callable[[np.ndarray, optuna.Trial], np.ndarray] = exec_context.get("evaluate")
                if not process_func:
                    raise ValueError("Agent 代码中未定义 evaluate 函数")

                sig = inspect.signature(process_func)
                sig.bind(
                    np.random.randint(0, 256, (48, 64), dtype=np.uint8),
                )

                self._eva_func = process_func
                self._eva_code = hash(code_str)
                return
            except SecurityViolation:
                raise
            except TypeError:
                raise
            except SyntaxError:
                raise
            except Exception as e:
                if self._recover_missing_module_dependency(e):
                    continue
                raise
        raise RuntimeError("Failed to prepare evaluate code after dependency auto-recovery")
    
    def prepare_code(self, code_str: str):
        for _ in range(2):
            exec_context = self._base_namespace.copy()
            cv_wrappers_obj = SimpleNamespace()

            for tool_name, tool_data in self.registry._tools.items():
                exec_context[tool_name] = tool_data["func"]
                setattr(cv_wrappers_obj, tool_name, tool_data["func"])

            exec_context["cv_wrappers"] = cv_wrappers_obj

            try:
                astree = ast.parse(code_str)
                checker = AgentCodeChecker(self._build_allowed_import_prefixes())
                checker.visit(astree)

                exec(code_str, exec_context)

                process_func: Callable[[np.ndarray, optuna.Trial], np.ndarray] = exec_context.get("process")
                if not process_func:
                    raise ValueError("Agent 代码中未定义 process 函数")

                sig = inspect.signature(process_func)
                self._func_accepts_cache = True
                try:
                    sig.bind(
                        np.random.randint(0, 256, (48, 64), dtype=np.uint8),
                        optuna.trial.FixedTrial({}),
                        dict()
                    )
                except TypeError:
                    sig.bind(
                        np.random.randint(0, 256, (48, 64), dtype=np.uint8),
                        optuna.trial.FixedTrial({})
                    )
                    self._func_accepts_cache = False

                self._func = process_func
                self._code = hash(code_str)
                return
            except SecurityViolation:
                raise
            except TypeError:
                raise
            except SyntaxError:
                raise
            except Exception as e:
                if self._recover_missing_module_dependency(e):
                    continue
                raise
        raise RuntimeError("Failed to prepare process code after dependency auto-recovery")

    def test_generated_tools(self, code_str: str, func_name: str, schema: dict) -> Exception | None:
        while True:
            exec_context = self._base_namespace.copy()
            try:
                astree = ast.parse(code_str)
                checker = AgentCodeChecker(self._build_allowed_import_prefixes())
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
                if self._recover_missing_module_dependency(e):
                    continue
                return e
        # return RuntimeError("Failed to test tool after dependency auto-recovery")

    def execute_pipeline(
        self,
        code_str: str,
        img: np.ndarray,
        trial: optuna.Trial,
        cache: dict | None = None
    ) -> np.ndarray | Exception:
        """
        使用 exec() 执行 code_str。

        要求 code_str 必须定义了一个 process(img: np.ndarray, trial: optuna.Trial) -> nd.ndarray 函数。

        返回处理后的图像。如果报错，抛出携带详细 Traceback 的异常供 LLM 修复。
        """
        # 创建一个命名空间，注册基础组件
        if not self._func or self._code != hash(code_str):
            self.prepare_code(code_str)

        runtime_cache = cache if cache is not None else {}
        self._ensure_runtime_meta(runtime_cache)

        last_error: Exception | None = None
        for _ in range(6):
            adaptive_trial = SandboxExecutor._AdaptiveTrial(self, trial, runtime_cache)
            try:
                if self._func_accepts_cache:
                    result = self._func(img, adaptive_trial, runtime_cache)
                else:
                    result = self._func(img, adaptive_trial)
                runtime_cache["__runtime_param_lock__"].update(adaptive_trial.pending_locks)
                return result
            except Exception as e:
                last_error = e
                if self._recover_missing_module_dependency(e):
                    continue
                if self._is_device_error(e):
                    self.preferred_device = "cpu"
                    runtime_cache.setdefault("__runtime__", {})["preferred_device"] = "cpu"
                    continue
                if not self._is_oom_error(e):
                    raise
                self._release_device_memory()
                if not self._next_oom_level(runtime_cache):
                    break

        if last_error:
            raise RuntimeError(
                f"Process failed after OOM auto-fallback (level={runtime_cache.get('__runtime_oom_level__', 0)}): {last_error}"
            ) from last_error
        raise RuntimeError("Process failed with unknown error")
    
    def execute_evaluate(self, code_str: str, img: np.ndarray, evaluator: Evaluator) -> float:
        """
        使用 exec() 执行 code_str。

        要求 code_str 必须定义了一个 evaluate(img: np.ndarray) -> nd.ndarray 函数。

        返回处理后的图像。如果报错，抛出携带详细 Traceback 的异常供 LLM 修复。
        """
        # 创建一个命名空间，注册基础组件
        if not self._eva_func or self._eva_code != hash(code_str):
            self.prepare_evaluate_code(code_str, evaluator)

        return self._eva_func(img)
    
    def execute_pipeline_direct(
        self,
        code_str: str,
        img: np.ndarray,
        params: dict,
        cache: dict | None = None
    ) -> np.ndarray:
        """
        通过注入 Fake Trial 实现参数复用。
        """
        if not self._func or self._code != hash(code_str):
            self.prepare_code(code_str)
            
        # 实例化伪造的 Trial
        fake_trial = optuna.trial.FixedTrial(params)

        runtime_cache = cache if cache is not None else {}
        self._ensure_runtime_meta(runtime_cache)
        last_error: Exception | None = None
        for _ in range(6):
            adaptive_trial = SandboxExecutor._AdaptiveTrial(self, fake_trial, runtime_cache)
            try:
                if self._func_accepts_cache:
                    result = self._func(img, adaptive_trial, runtime_cache)
                else:
                    result = self._func(img, adaptive_trial)
                runtime_cache["__runtime_param_lock__"].update(adaptive_trial.pending_locks)
                return result
            except Exception as e:
                last_error = e
                if self._recover_missing_module_dependency(e):
                    continue
                if self._is_device_error(e):
                    self.preferred_device = "cpu"
                    runtime_cache.setdefault("__runtime__", {})["preferred_device"] = "cpu"
                    continue
                if not self._is_oom_error(e):
                    raise
                self._release_device_memory()
                if not self._next_oom_level(runtime_cache):
                    break

        if last_error:
            raise RuntimeError(
                f"Direct process failed after OOM auto-fallback (level={runtime_cache.get('__runtime_oom_level__', 0)}): {last_error}"
            ) from last_error
        raise RuntimeError("Direct process failed with unknown error")
