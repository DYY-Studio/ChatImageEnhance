import numpy as np
import inspect
import json
import yaml

from typing import Callable

class ToolRegistry:
    """
    管理可用算子，将其转换为大模型可理解的 JSON Schema

    供 Planner 和 Coder 生成代码时参考。
    """
    def __init__(self):
        self._tools = {}

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
        for name, param in sig.parameters.items():
            if name not in ['img']: # 排除默认参数
                parameters_schema[name] = {"type": str(param.annotation)}
                
        self._tools[name if name else func.__name__] = {
            "func": func,
            "schema": {
                "name": name if name else func.__name__,
                "description": doc,
                "parameters": parameters_schema
            }
        }

    def dynamic_register(self, func: Callable, schema: dict):
        """动态注册LLM生成的算子"""
        func_name = schema["name"]
        
        # 2. 注册到内存
        self._tools[func_name] = {
            "func": func,
            "schema": schema,
            "is_dynamic": True # 标记为动态生成的工具
        }
        
        # 3. 持久化（可选）：将 code_str 写入到 tools/custom_wrappers.py 
        # 以便下次启动时自动加载
        # self._persist_to_file(code_str, schema)

    def register(self, name: str, func: Callable, description: str, params_schema: dict):
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
        """
        self._tools[name] = {
            "func": func,  # 供本地 Python 真正执行的函数指针
            "schema": {    # 供大模型阅读的说明书
                "name": name,
                "description": description,
                "parameters": params_schema
            }
        }

    def get_all_schemas_for_llm(self) -> str:
        """返回所有算子的规范说明（注入到 Prompt 中）"""
        return yaml.dump(
            [tool["schema"] for tool in self._tools.values()], 
            indent=2,
            allow_unicode=True
        )
    
    def get_all_schemas_for_llm_short(self) -> str:
        """返回所有算子的规范说明（不带参数）（注入到 Prompt 中）"""
        return yaml.dump(
            [{
                "name": tool["schema"]["name"],
                "description": tool["schema"]["description"],
            } for tool in self._tools.values()], 
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