import numpy as np
import inspect
import yaml
import importlib.util
import sys
import logging
import re
import functools

from pathlib import Path
from typing import Callable, Literal
from utils import get_executable_dir

logger = logging.getLogger("ToolRegistry")

class ToolRegistry:
    """
    管理可用算子，将其转换为大模型可理解的 JSON Schema

    供 Planner 和 Coder 生成代码时参考。
    """
    def __init__(self):
        self._tools = {}
        self._last_custom_tool_errors: dict[str, str] = {}

    @property
    def last_custom_tool_errors(self):
        return self._last_custom_tool_errors.copy()

    @staticmethod
    def _safe_repo_cache_name(repo_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(repo_id or "").strip())

    @classmethod
    def resolve_model_dir(cls, schema: dict | None) -> str:
        """
        Resolve the local model directory for a tool schema.

        Dynamic deep-learning tools may come from Hugging Face, ModelScope, or
        GitHub. Their cache roots differ, so callers should not hardcode a
        generic caches/model_assets path.
        """
        if not isinstance(schema, dict):
            return ""

        source = str(schema.get("model_source") or schema.get("source") or "").strip().lower()
        repo_id = str(schema.get("model_repo_id") or schema.get("repo_id") or "").strip()
        if source in {"github", "huggingface", "modelscope"}:
            base = get_executable_dir() / "caches" / "model_assets" / source
            if repo_id:
                return str((base / cls._safe_repo_cache_name(repo_id)).resolve())
            return str(base.resolve())

        # Backward-compatible fallback for older local tools. New schemas should
        # store source/repo_id instead of absolute paths so app relocation works.
        for key in ("model_dir", "download_dir", "model_download_dir", "model_asset_dir"):
            value = schema.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            path = get_executable_dir() / value.strip()
            return str(path.resolve()) if not Path(value.strip()).is_absolute() else value.strip()

        return ""

    @staticmethod
    def _accepts_keyword(sig: inspect.Signature, name: str) -> bool:
        if name in sig.parameters:
            return sig.parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

    @classmethod
    def build_runtime_kwargs(
        cls,
        func: Callable,
        schema: dict | None,
        kwargs: dict | None = None,
        filter_unknown: bool = True
    ) -> dict:
        """
        Return a copy of kwargs with runtime-owned defaults injected.

        Currently this injects model_dir from schema metadata. Caller supplied
        non-empty model_dir always wins.
        """
        final_kwargs = dict(kwargs or {})
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return final_kwargs

        if cls._accepts_keyword(sig, "model_dir"):
            current = final_kwargs.get("model_dir")
            if current is None or (isinstance(current, str) and not current.strip()):
                model_dir = cls.resolve_model_dir(schema)
                if model_dir:
                    final_kwargs["model_dir"] = model_dir

        if not filter_unknown:
            return final_kwargs

        return {
            key: value
            for key, value in final_kwargs.items()
            if cls._accepts_keyword(sig, key)
        }

    @classmethod
    def _wrap_tool_func(cls, func: Callable, schema: dict) -> Callable:
        @functools.wraps(func)
        def wrapped(img: np.ndarray, *args, **kwargs):
            try:
                bound = inspect.signature(func).bind_partial(img, *args, **kwargs)
                has_model_dir = "model_dir" in bound.arguments
                model_dir_value = bound.arguments.get("model_dir")
                model_dir_blank = (
                    model_dir_value is None
                    or (isinstance(model_dir_value, str) and not model_dir_value.strip())
                )
            except TypeError:
                has_model_dir = "model_dir" in kwargs
                model_dir_value = kwargs.get("model_dir")
                model_dir_blank = (
                    model_dir_value is None
                    or (isinstance(model_dir_value, str) and not model_dir_value.strip())
                )

            if (not has_model_dir) or ("model_dir" in kwargs and model_dir_blank):
                kwargs = cls.build_runtime_kwargs(func, schema, kwargs, filter_unknown=False)
            return func(img, *args, **kwargs)

        return wrapped

    @staticmethod
    def _normalize_schema_filter_mode(
        allow_learning: bool = True,
        learning_mode: str | None = None
    ) -> Literal["all", "traditional_only", "learning_only"]:
        if learning_mode is None:
            return "all" if allow_learning else "traditional_only"

        mode = str(learning_mode).strip().lower()
        alias = {
            "all": "all",
            "both": "all",
            "traditional_only": "traditional_only",
            "only_traditional": "traditional_only",
            "traditional": "traditional_only",
            "learning_only": "learning_only",
            "only_learning": "learning_only",
            "only_deep_learning": "learning_only",
            "deep_learning_only": "learning_only",
            "deep_only": "learning_only",
        }
        if mode not in alias:
            logger.warning(f"未知 schema 过滤模式: {learning_mode}，回落到 all")
            return "all"
        normalized = alias[mode]
        if not allow_learning:
            return "traditional_only"
        return normalized

    @staticmethod
    def sanitize_tool_name(name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            raise ValueError("工具名不能为空")
        if "/" in raw or "\\" in raw:
            raise ValueError("工具名不能包含路径分隔符")
        if raw != raw.strip(". "):
            raise ValueError("工具名不能以点号或空格开头/结尾")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
            raise ValueError("工具名必须是合法的 Python 标识符（字母/数字/下划线）")

        reserved_names = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(0, 10)),
            *(f"LPT{i}" for i in range(0, 10))
        }
        if raw.upper() in reserved_names:
            raise ValueError(f"工具名 {raw} 为系统保留名称")
        return raw

    @staticmethod
    def get_custom_tool_paths(name: str):
        safe_name = ToolRegistry.sanitize_tool_name(name)
        custom_tool_dir = get_executable_dir() / "tools/custom"
        base = custom_tool_dir / safe_name

        custom_dir_resolved = custom_tool_dir.resolve()
        base_resolved = base.resolve()
        if base_resolved.parent != custom_dir_resolved:
            raise ValueError("工具路径越界，已拒绝写入/读取")

        return safe_name, custom_tool_dir, base

    def load_custom_tool(self, name: str):
        try:
            safe_name, _, base = ToolRegistry.get_custom_tool_paths(name)
        except Exception as e:
            self._last_custom_tool_errors[str(name)] = str(e)
            logger.warning(f"Custom tool {name} invalid: {e}")
            return False, str(e)

        tool_py = base.with_suffix(".py")
        tool_yaml = base.with_suffix(".yaml")
        if not tool_py.exists() or not tool_yaml.exists():
            msg = f"工具文件缺失: {tool_py.name} 或 {tool_yaml.name}"
            self._last_custom_tool_errors[safe_name] = msg
            return False, msg
        
        try:
            schema = yaml.safe_load(tool_yaml.read_text('utf-8'))
            if not isinstance(schema, dict):
                raise ValueError("工具 Schema 必须是 JSON/YAML 对象")
            schema_name = ToolRegistry.sanitize_tool_name(schema.get("name", ""))
            if schema_name != safe_name:
                raise ValueError(
                    f"工具名不一致：文件名为 {safe_name}，Schema 名为 {schema_name}"
                )
            module = ToolRegistry._load_tool_from_file(
                f"dynamic_tools_{schema['name']}", str(tool_py.absolute())
            )
            self.dynamic_register(
                getattr(module, schema['name']),
                schema
            )
            self._last_custom_tool_errors.pop(safe_name, None)
            logger.info(f"Successfully load custom tool: {schema['name']}")
            return True, None
        except Exception as e:
            self._last_custom_tool_errors[safe_name] = str(e)
            logger.info(f"Custom tool {name} dynamic load failed: {e}")
            return False, str(e)

    def load_custom_tools(self):
        custom_tool_dir = get_executable_dir() / "tools/custom"
        if not custom_tool_dir.exists():
            return
        
        for file in custom_tool_dir.iterdir():
            if not file.is_file() or file.suffix.lower() != '.py':
                continue

            self.load_custom_tool(file.with_suffix('').name)


    @staticmethod
    def _load_tool_from_file(module_name: str, file_path: str):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @property
    def tools(self):
        return self._tools.copy()

    def auto_register(self, func: Callable, name: str | None = None):
        """
        传入函数指针，自动生成说明书

        注意：使用该函数时，注册的函数必须要有完善的Docstring和类型注解
        """
        doc = inspect.getdoc(func) # 
        sig = inspect.signature(func)
        
        # 解析参数类型，自动构建 JSON Schema
        parameters_schema = {}
        for param_name, param in sig.parameters.items():
            if param_name not in ['img']: # 排除默认参数
                parameters_schema[param_name] = {"type": str(param.annotation)}

        tool_name = name if name else func.__name__
        schema = {
            "name": tool_name,
            "description": doc,
            "parameters": parameters_schema,
        }
        self._tools[tool_name] = {
            "func": self._wrap_tool_func(func, schema),
            "raw_func": func,
            "schema": schema
        }

    def dynamic_register(self, func: Callable, schema: dict, performance: str = 'unknown'):
        """动态注册LLM生成的算子"""
        func_name = ToolRegistry.sanitize_tool_name(schema["name"])
        schema = dict(schema)
        schema["name"] = func_name
        schema["requires_learning"] = bool(schema.get("requires_learning", False))
        if performance != 'unknown':
            schema['cost'] = performance
        
        self._tools[func_name] = {
            "func": self._wrap_tool_func(func, schema),
            "raw_func": func,
            "schema": schema,
            "is_dynamic": True # 标记为动态生成的工具
        }

    def dynamic_unregister(self, name: str) -> dict | None:
        """将动态注册算子注销/卸载"""
        if name in self._tools:
            if self._tools[name].get('is_dynamic', False):
                return self._tools.pop(name)
        
        return None

    def register(self, 
        name: str, 
        func: Callable, 
        description: str, 
        params_schema: dict,
        performance: Literal['very fast', 'faster', 'fast', 'medium', 'slow', 'slower', 'very slow', 'slowest'],
        requires_learning: bool = False
    ):
        """
        注册一个 CV 函数及其参数范围
        
        :param name: 函数名称
        :param func: CV函数指针
        :param description: 给LLM解释该函数
        :param params_schema: 该函数的参数名称，及其取值范围或可取参数
            
            应当为以下格式或类似的兼容性格式
            ```
            { 
                "<param_name>": {
                    "type": "<参数变量类型>",
                    "range": <取值范围，或可取的参数列表>,
                    "description": "<解释参数作用>"
                }
            }
            ```
        :param performance: 描述函数的运行速度
        """
        schema = {    # 供大模型阅读的说明书
            "name": name,
            "description": description,
            "parameters": params_schema,
            "cost": performance,
            "requires_learning": requires_learning
        }
        self._tools[name] = {
            "func": self._wrap_tool_func(func, schema),  # 供本地 Python 真正执行的函数指针
            "raw_func": func,
            "schema": schema,
        }

    def _filter_tools_for_llm(
        self,
        allow_learning: bool = True,
        learning_mode: str | None = None
    ) -> list[dict]:
        mode = self._normalize_schema_filter_mode(allow_learning, learning_mode)
        filtered = []
        for tool in self._tools.values():
            requires_learning = bool(tool["schema"].get("requires_learning", False))
            if mode == "traditional_only" and requires_learning:
                continue
            if mode == "learning_only" and not requires_learning:
                continue
            filtered.append(tool)
        return filtered

    def get_schemas_for_llm(
        self,
        allow_learning: bool = True,
        learning_mode: str | None = None
    ) -> list[dict]:
        return [
            tool["schema"] for tool in self._filter_tools_for_llm(
                allow_learning=allow_learning,
                learning_mode=learning_mode
            )
        ]

    def get_all_schemas_for_llm(
        self,
        allow_learning: bool = True,
        learning_mode: str | None = None
    ) -> str:
        """返回所有算子的规范说明（注入到 Prompt 中）"""
        schemas = self.get_schemas_for_llm(
            allow_learning=allow_learning,
            learning_mode=learning_mode
        )
        return yaml.dump(
            schemas, 
            indent=2,
            allow_unicode=True
        )
    
    def get_all_schemas_for_llm_short(
        self,
        allow_learning: bool = True,
        learning_mode: str | None = None
    ) -> str:
        """返回所有算子的规范说明（不带参数）（注入到 Prompt 中）"""
        tool_list = self._filter_tools_for_llm(
            allow_learning=allow_learning,
            learning_mode=learning_mode
        )
        return yaml.dump(
            [{
                "name": tool["schema"]["name"],
                "description": tool["schema"]["description"],
            } for tool in tool_list], 
            indent=2,
            allow_unicode=True
        )

    def execute_tool(self, name: str, img: np.ndarray, **kwargs) -> np.ndarray:
        """
        （拓展用占位）本地执行调用时使用
        """
        if name not in self._tools:
            raise ValueError(f"未找到算子: {name}")
        func = self._tools[name]["func"] # 取出函数指针
        return func(img, **kwargs) # 注入参数
