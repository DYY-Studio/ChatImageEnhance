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

MODEL_OFFLINE_LOADER_CALLS = {
    "from_pretrained",
    "pipeline",
    "Pipeline",
    "snapshot_download",
    "hf_hub_download",
}

MODEL_OFFLINE_MODULE_ROOTS = {
    "transformers",
    "diffusers",
    "modelscope",
    "huggingface_hub",
}

MODEL_OFFLINE_NESTED_KWARGS = {
    "model_kwargs",
    "tokenizer_kwargs",
    "processor_kwargs",
    "config_kwargs",
    "pipeline_kwargs",
}


def _call_name(func: ast.AST) -> str:
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    if not parts:
        return "<call>"
    return ".".join(reversed(parts))


def _is_true_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    return None


def _mapping_has_true_key(node: ast.AST, key: str) -> bool:
    if isinstance(node, ast.Dict):
        for item_key, item_value in zip(node.keys, node.values):
            if item_key is not None and _string_literal(item_key) == key and _is_true_literal(item_value):
                return True
        return False

    if isinstance(node, ast.Call) and _call_name(node.func).split(".")[-1] == "dict":
        return any(
            kw.arg == key and _is_true_literal(kw.value)
            for kw in node.keywords
        )

    return False


def _call_has_local_files_only_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "local_files_only" and _is_true_literal(kw.value):
            return True
        if kw.arg in MODEL_OFFLINE_NESTED_KWARGS and _mapping_has_true_key(kw.value, "local_files_only"):
            return True
        if kw.arg is None and _mapping_has_true_key(kw.value, "local_files_only"):
            return True
    return False


def _collect_model_loader_import_context(tree: ast.AST) -> tuple[set[str], set[str]]:
    module_aliases: set[str] = set()
    loader_aliases: set[str] = set()
    direct_loader_names = MODEL_OFFLINE_LOADER_CALLS - {"from_pretrained"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name or "").split(".", maxsplit=1)[0]
                if root in MODEL_OFFLINE_MODULE_ROOTS:
                    module_aliases.add(alias.asname or root)
            continue

        if isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", maxsplit=1)[0]
            if root not in MODEL_OFFLINE_MODULE_ROOTS:
                continue
            module_aliases.add(root)
            for alias in node.names:
                if alias.name in direct_loader_names:
                    loader_aliases.add(alias.asname or alias.name)

    return module_aliases, loader_aliases


def _is_model_loader_call(name: str, module_aliases: set[str], loader_aliases: set[str]) -> bool:
    parts = name.split(".")
    leaf = parts[-1]
    if leaf == "from_pretrained":
        return True
    if len(parts) == 1 and name in loader_aliases:
        return True
    if leaf not in MODEL_OFFLINE_LOADER_CALLS:
        return False
    return parts[0] in module_aliases or parts[0] in MODEL_OFFLINE_MODULE_ROOTS


def find_model_loader_calls_missing_local_files_only(code_str: str) -> list[tuple[int, str]]:
    tree = ast.parse(code_str)
    module_aliases, loader_aliases = _collect_model_loader_import_context(tree)
    missing: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if not _is_model_loader_call(name, module_aliases, loader_aliases):
            continue
        if not _call_has_local_files_only_true(node):
            missing.append((getattr(node, "lineno", 0), name))
    return missing


def validate_model_loaders_local_files_only(code_str: str):
    missing = find_model_loader_calls_missing_local_files_only(code_str)
    if not missing:
        return

    details = ", ".join(
        f"{name} at line {line}" if line else name
        for line, name in missing
    )
    raise ValueError(
        "Hugging Face / ModelScope model loader calls must include "
        f"`local_files_only=True`: {details}"
    )

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
