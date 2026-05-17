import ast
from typing import Iterable

class SecurityViolation(Exception):
    """自定义安全违规异常"""
    pass

DEFAULT_ALLOWED_IMPORT_PREFIXES = (
    "numpy",
    "cv2",
    "skimage",
    "PIL",
    "torch",
    "torchvision",
    "modelscope",
    "transformers",
    "diffusers",
    "scipy",
    "einops",
    "safetensors",
    "math",
    "typing",
    "collections",
    "functools",
    "itertools",
)

def is_allowed_import_path(module_name: str, allowed_prefixes: Iterable[str]) -> bool:
    name = str(module_name or "").strip()
    if not name:
        return False
    for prefix in allowed_prefixes:
        normalized = str(prefix or "").strip()
        if not normalized:
            continue
        if name == normalized or name.startswith(normalized + "."):
            return True
    return False

class AgentCodeChecker(ast.NodeVisitor):
    def __init__(self, allowed_import_prefixes: Iterable[str] | None = None):
        # 定义黑名单：禁止直接调用的内置函数
        self.forbidden_names = {
            'eval', 'exec', 'getattr', 'setattr', 'delattr', 'input', 'open',
            '__import__', '__builtins__', 'compile', 'breakpoint',
            'globals', 'locals', 'vars'
        }
        prefixes = tuple(allowed_import_prefixes or DEFAULT_ALLOWED_IMPORT_PREFIXES)
        self.allowed_import_prefixes = tuple(p.strip() for p in prefixes if str(p).strip())
        self._class_depth = 0

    def _check_import_target(self, module_name: str):
        if not is_allowed_import_path(module_name, self.allowed_import_prefixes):
            raise SecurityViolation(f"禁止导入非白名单模块: {module_name}")

    @staticmethod
    def _is_super_init_attribute(node: ast.Attribute) -> bool:
        if node.attr != "__init__":
            return False
        value = node.value
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "super"
        )

    def visit_ClassDef(self, node):
        if node.name.startswith("__"):
            raise SecurityViolation(f"禁止定义私有类: {node.name}")
        self._class_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._class_depth -= 1

    def visit_FunctionDef(self, node):
        if node.name.startswith("__") and not (self._class_depth > 0 and node.name == "__init__"):
            raise SecurityViolation(f"禁止定义私有函数: {node.name}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        raise SecurityViolation("禁止定义 async 函数")
        
    def visit_Import(self, node):
        for alias in node.names:
            self._check_import_target(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.level and node.level > 0:
            raise SecurityViolation("禁止使用相对导入")

        module_name = str(node.module or "").strip()
        self._check_import_target(module_name)

        for alias in node.names:
            if alias.name == "*":
                raise SecurityViolation("禁止使用 from ... import *")
            if alias.name.startswith("__"):
                raise SecurityViolation(f"禁止导入私有符号: {alias.name}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # 拦截：访问 __subclasses__ 等双下划线私有属性（Python 逃逸常用手段）
        if node.attr.startswith('__') and not self._is_super_init_attribute(node):
            raise SecurityViolation(f"禁止访问私有属性: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        # 拦截：直接调用黑名单中的危险函数
        if node.id in self.forbidden_names:
            raise SecurityViolation(f"禁止使用危险函数: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # 对特定函数调用频率或参数的逻辑检查（占位）
        self.generic_visit(node)
