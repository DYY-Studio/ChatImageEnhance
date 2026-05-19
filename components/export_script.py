"""Standalone processing-script export helpers.

This module owns the AST-based dependency discovery and script generation used
by the chat history export button. Keep Streamlit rendering code in
components.__init__ and export-specific code here.
"""

import ast
import builtins
import inspect
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import Callable

import streamlit as st

from tools import global_registry
from utils import get_executable_dir

logger = logging.getLogger("ExportScript")


def _normalize_import_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []

    normalized: list[str] = []
    for item in values:
        token = str(item or "").strip()
        if not token:
            continue
        if token in normalized:
            continue
        normalized.append(token)
    return normalized

def _package_to_import_name(package: str) -> str | None:
    token = str(package or "").strip()
    if not token:
        return None
    token = token.split(";", maxsplit=1)[0].strip()
    token = token.split("[", maxsplit=1)[0].strip()
    token = re.split(r"(==|!=|>=|<=|>|<|~=)", token, maxsplit=1)[0].strip()
    if not token:
        return None

    alias = {
        "pillow": "PIL",
        "opencv-python": "cv2",
        "opencv-contrib-python": "cv2",
        "opencv-python-headless": "cv2",
        "opencv-contrib-python-headless": "cv2",
        "scikit-image": "skimage",
        "pyyaml": "yaml",
        "huggingface-hub": "huggingface_hub",
        "python-dateutil": "dateutil",
        "beautifulsoup4": "bs4",
    }
    canonical = re.sub(r"[-_.]+", "-", token.lower())
    if canonical in alias:
        return alias[canonical]

    return token.replace("-", "_")

def _extract_wrapper_source(func: Callable) -> str:
    """
    从工具模块或类中提取指定函数的源代码，并格式化为静态方法
    
    Args:
        wrapper_module: 工具模块或类（如 cv_wrappers 模块）
        func_name: 函数名
    
    Returns:
        格式化后的静态方法源代码字符串
    """
    try:
        # 获取函数对象（支持模块和类两种情况）
        if func is None:
            return ""
        
        # 使用 inspect 获取源代码
        source = inspect.getsource(func)
        
        # 清理缩进并转换为静态方法格式
        lines = source.split('\n')
        if lines:
            # 找到最小非空行的缩进
            min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
            
            # 处理第一行（函数定义行），添加 @staticmethod 装饰器
            cleaned_lines = ['    @staticmethod']
            
            for i, line in enumerate(lines):
                if line.strip():
                    # 移除原有缩进并添加标准缩进（4空格）
                    cleaned_line = '    ' + line[min_indent:]
                    
                    # 如果是第一行且是 def 开头，保持原样（已经添加了装饰器）
                    if i == 0 and line.strip().startswith('def'):
                        cleaned_lines.append(cleaned_line)
                    else:
                        cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append('')
            
            return '\n'.join(cleaned_lines)
        return source
    except Exception as e:
        print(f"Warning: Failed to extract source for {func.__name__}: {e}")
        return ""

def _extract_used_operator_names(process_code: str) -> list[str]:
    code = str(process_code or "")
    found: list[tuple[int, str]] = []

    for match in re.finditer(r"\bcv_wrappers\.(\w+)\s*\(", code):
        found.append((match.start(), match.group(1)))

    for tool_name in global_registry.tools:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(tool_name)):
            continue
        match = re.search(rf"\b{re.escape(tool_name)}\s*\(", code)
        if match:
            found.append((match.start(), str(tool_name)))

    operators: list[str] = []
    for _, name in sorted(found, key=lambda item: item[0]):
        if name not in operators:
            operators.append(name)
    return operators

def _strip_markdown_code_fence(code: str) -> str:
    text = str(code or "").strip()
    match = re.fullmatch(r"```(?:python)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text

def _message_has_exportable_process(msg: dict) -> bool:
    return bool(str(msg.get("process_code") or "").strip()) and "best_params" in msg

def _previous_exportable_assistant_index(messages: list[dict], before_index: int) -> int | None:
    for idx in range(before_index - 1, -1, -1):
        msg = messages[idx]
        if (
            msg.get("role") == "assistant"
            and "image" in msg
            and not msg.get("test_mode")
            and _message_has_exportable_process(msg)
        ):
            return idx
    return None

def _collect_export_process_chain(messages: list[dict], current_index: int) -> list[tuple[int, dict]]:
    if current_index < 0 or current_index >= len(messages):
        return []
    current = messages[current_index]
    if (
        current.get("role") != "assistant"
        or "image" not in current
        or current.get("test_mode")
        or not _message_has_exportable_process(current)
    ):
        return []

    chain: list[tuple[int, dict]] = [(current_index, current)]
    cursor = current_index
    while messages[cursor].get("input_from_previous") or messages[cursor].get("input_source") == "previous_result":
        prev_idx = _previous_exportable_assistant_index(messages, cursor)
        if prev_idx is None:
            break
        chain.append((prev_idx, messages[prev_idx]))
        cursor = prev_idx

    return list(reversed(chain))

def _params_lookup_source(param_name: str) -> str:
    escaped = param_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'params["{escaped}"]'

def _trial_suggest_param_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Attribute):
        return None
    if not call.func.attr.startswith("suggest_"):
        return None
    if not isinstance(call.func.value, ast.Name) or call.func.value.id != "trial":
        return None

    if call.args:
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value

    for keyword in call.keywords:
        if keyword.arg in {"name", "param_name"} and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None

def _replace_ast_source_segments(source: str, replacements: list[tuple[ast.AST, str]]) -> str:
    if not replacements:
        return source

    line_offsets: list[int] = []
    current_offset = 0
    for line in source.splitlines(keepends=True):
        line_offsets.append(current_offset)
        current_offset += len(line.encode("utf-8"))
    if not line_offsets:
        line_offsets.append(0)

    source_bytes = source.encode("utf-8")
    ranges: list[tuple[int, int, bytes]] = []
    for node, replacement in replacements:
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        end_lineno = getattr(node, "end_lineno", None)
        end_col_offset = getattr(node, "end_col_offset", None)
        if None in {lineno, col_offset, end_lineno, end_col_offset}:
            continue
        if lineno < 1 or end_lineno < 1:
            continue
        if lineno > len(line_offsets) or end_lineno > len(line_offsets):
            continue

        start = line_offsets[lineno - 1] + col_offset
        end = line_offsets[end_lineno - 1] + end_col_offset
        if 0 <= start <= end <= len(source_bytes):
            ranges.append((start, end, replacement.encode("utf-8")))

    if not ranges:
        return source

    for start, end, replacement in sorted(ranges, key=lambda item: item[0], reverse=True):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    return source_bytes.decode("utf-8")

def _replace_trial_suggest_calls(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return re.sub(
            r"trial\.suggest_[a-zA-Z_]+\(\s*[\"']([a-zA-Z0-9_]+)[\"'][\s\S]*?\)",
            lambda match: _params_lookup_source(match.group(1)),
            source,
        )

    replacements: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        param_name = _trial_suggest_param_name(node)
        if param_name is not None:
            replacements.append((node, _params_lookup_source(param_name)))
    return _replace_ast_source_segments(source, replacements)

def _process_code_to_export_step(process_code: str, function_name: str) -> str:
    cleaned_code = _strip_markdown_code_fence(process_code)

    try:
        tree = ast.parse(cleaned_code)
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process":
                func_node = node
                break

        if func_node and func_node.body:
            start_line = func_node.body[0].lineno - 1
            end_line = func_node.end_lineno
            lines = cleaned_code.split("\n")[start_line:end_line]
            non_empty = [line for line in lines if line.strip()]
            min_indent = min((len(line) - len(line.lstrip()) for line in non_empty), default=0)
            normalized_lines = [
                line[min_indent:] if len(line) > min_indent else line
                for line in lines
            ]
            final_func_code = (
                f"def {function_name}(img, params, cache):\n"
                + "\n".join(["    " + line for line in normalized_lines])
            )
        else:
            final_func_code = cleaned_code
    except Exception:
        final_func_code = cleaned_code

    final_func_code = re.sub(r"\s*trial\s*=\s*.*?\n", "\n", final_func_code)
    return _replace_trial_suggest_calls(final_func_code)

_EXPORT_IMPORT_PACKAGES = {
    "cv2": "opencv-contrib-python",
    "numpy": "numpy",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "scipy": "scipy",
    "torch": "torch",
    "torchvision": "torchvision",
    "modelscope": "modelscope",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "diffusers": "diffusers",
    "open_clip": "open-clip-torch",
    "lpips": "lpips",
}

_LOCAL_EXPORT_MODULE_ROOTS = {
    "agents", "components", "constants", "core", "memory",
    "models", "sandbox", "tools", "utils",
}

_MODEL_FILE_SUFFIXES = (
    ".pth", ".pt", ".onnx", ".safetensors", ".ckpt", ".bin", ".model",
)

def _read_project_source(rel_path: str) -> str:
    path = get_executable_dir() / rel_path
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("导出脚本依赖源码读取失败: %s (%s)", rel_path, e)
        return ""

def _is_stdlib_import(import_name: str) -> bool:
    root = str(import_name or "").split(".", maxsplit=1)[0]
    if not root:
        return True
    if root in {"cv2", "numpy", "PIL", "skimage", "scipy", "torch", "torchvision"}:
        return False
    stdlib = getattr(sys, "stdlib_module_names", set())
    return root in stdlib or root in {"builtins", "__future__"}

def _is_local_project_module(module_name: str | None) -> bool:
    if not module_name:
        return False
    return module_name.split(".", maxsplit=1)[0] in _LOCAL_EXPORT_MODULE_ROOTS

def _module_name_to_rel_path(module_name: str) -> str | None:
    if not _is_local_project_module(module_name):
        return None
    rel_path = Path(*module_name.split(".")).with_suffix(".py")
    if (get_executable_dir() / rel_path).exists():
        return rel_path.as_posix()
    package_init = Path(*module_name.split(".")) / "__init__.py"
    if (get_executable_dir() / package_init).exists():
        return package_init.as_posix()
    return None

def _safe_source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment
    lines = source.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])

def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()

def _collect_bound_names(node: ast.AST) -> set[str]:
    bound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if child is not node:
                bound.add(child.name)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                bound.update(_target_names(target))
        elif isinstance(child, ast.AnnAssign):
            bound.update(_target_names(child.target))
        elif isinstance(child, ast.AugAssign):
            bound.update(_target_names(child.target))
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            bound.update(_target_names(child.target))
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            for item in child.items:
                if item.optional_vars is not None:
                    bound.update(_target_names(item.optional_vars))
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, ast.Import):
            for alias in child.names:
                bound.add(alias.asname or alias.name.split(".", maxsplit=1)[0])
        elif isinstance(child, ast.ImportFrom):
            for alias in child.names:
                if alias.name == "*":
                    continue
                bound.add(alias.asname or alias.name)
    return bound

def _collect_global_load_names(source: str) -> set[str]:
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", source))

    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    bound = _collect_bound_names(tree)
    builtin_names = set(dir(builtins))
    return {
        name for name in loaded - bound - builtin_names
        if not name.startswith("__")
    }

def _collect_module_symbols(source: str) -> tuple[dict[str, ast.AST], dict[str, ast.AST], dict[str, tuple[str | None, str | None, str]]]:
    definitions: dict[str, ast.AST] = {}
    assignments: dict[str, ast.AST] = {}
    imports: dict[str, tuple[str | None, str | None, str]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return definitions, assignments, imports

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    assignments[name] = node
        elif isinstance(node, ast.AnnAssign):
            for name in _target_names(node.target):
                assignments[name] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            line = _safe_source_segment(source, node)
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                imports.setdefault(bound_name, (alias.name, None, line))
        elif isinstance(node, ast.ImportFrom):
            line = _safe_source_segment(source, node)
            module_name = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound_name = alias.asname or alias.name
                imports.setdefault(bound_name, (module_name, alias.name, line))
    return definitions, assignments, imports

def _collect_import_nodes(source: str) -> list[ast.Import | ast.ImportFrom]:
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

def _strip_local_project_imports(source: str) -> str:
    nodes = _collect_import_nodes(source)
    remove_lines: set[int] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            is_local = any(_is_local_project_module(alias.name) for alias in node.names)
        else:
            is_local = _is_local_project_module(node.module)
        if not is_local:
            continue
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        remove_lines.update(range(start, end + 1))

    if not remove_lines:
        return source
    return "\n".join(
        line for idx, line in enumerate(source.splitlines(), start=1)
        if idx not in remove_lines
    )

def _normalize_additional_package_map(packages: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in _normalize_import_list(packages):
        import_name = _package_to_import_name(package)
        if import_name:
            result[import_name.split(".", maxsplit=1)[0]] = package
    return result

def _package_for_import(import_name: str, package_overrides: dict[str, str]) -> str:
    root = import_name.split(".", maxsplit=1)[0]
    if root in package_overrides:
        return package_overrides[root]
    if root in _EXPORT_IMPORT_PACKAGES:
        return _EXPORT_IMPORT_PACKAGES[root]
    return root

def _add_required_import(
    required_imports: dict[str, str],
    import_name: str,
    package_overrides: dict[str, str],
):
    root = str(import_name or "").split(".", maxsplit=1)[0]
    if not root or _is_stdlib_import(root) or _is_local_project_module(root):
        return
    required_imports.setdefault(root, _package_for_import(root, package_overrides))

def _add_import_context_from_source(
    source: str,
    import_lines: list[str],
    required_imports: dict[str, str],
    package_overrides: dict[str, str],
):
    source = textwrap.dedent(source)
    for node in _collect_import_nodes(source):
        line = _safe_source_segment(source, node).strip()
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            is_local = any(_is_local_project_module(name) for name in names)
            for name in names:
                _add_required_import(required_imports, name, package_overrides)
        else:
            names = [node.module or ""]
            is_local = _is_local_project_module(node.module)
            if node.module:
                _add_required_import(required_imports, node.module, package_overrides)
        if line and not is_local and line not in import_lines:
            import_lines.append(line)

def _collect_model_file_literals(source: str) -> list[str]:
    source = textwrap.dedent(source)
    model_files: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        candidates = re.findall(r"['\"]([^'\"]+)['\"]", source)
    else:
        candidates = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]

    mentions_model_dir = bool(re.search(r"\bmodel_dir\b|[\"']models[\"']", source))
    for value in candidates:
        token = value.replace("\\", "/").strip()
        lower = token.lower()
        if not lower.endswith(_MODEL_FILE_SUFFIXES):
            continue
        if token.startswith("models/"):
            rel_path = token
        elif "/" not in token and mentions_model_dir:
            rel_path = f"models/{token}"
        else:
            continue
        if rel_path not in model_files:
            model_files.append(rel_path)
    return model_files

def _export_schema_subset(schema: dict | None) -> dict:
    if not isinstance(schema, dict):
        return {}
    keys = (
        "name", "source", "model_source", "repo_id", "model_repo_id",
        "model_dir", "download_dir", "model_download_dir", "model_asset_dir",
        "require_files", "required_files", "downloaded_files",
        "additional_imports", "additional_packages",
    )
    return {
        key: schema[key]
        for key in keys
        if key in schema
    }

def _schema_model_files(schema: dict | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    values: list[str] = []
    for key in ("require_files", "required_files", "downloaded_files"):
        raw = schema.get(key)
        if isinstance(raw, str):
            raw_values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            raw_values = list(raw)
        else:
            raw_values = []
        for value in raw_values:
            token = str(value or "").replace("\\", "/").strip()
            if not token or not token.lower().endswith(_MODEL_FILE_SUFFIXES):
                continue
            if token.startswith("models/"):
                rel_path = token
            elif "/" not in token:
                rel_path = f"models/{token}"
            else:
                continue
            if rel_path not in values:
                values.append(rel_path)
    return values

def _schema_additional_import_lines(schema: dict | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    lines: list[str] = []
    for import_name in _normalize_import_list(schema.get("additional_imports")):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", import_name):
            line = f"import {import_name}"
            if line not in lines:
                lines.append(line)
    return lines

def _inline_local_module_symbols(
    module_name: str,
    names: set[str] | None,
    support_chunks: list[str],
    import_lines: list[str],
    required_imports: dict[str, str],
    required_model_files: list[str],
    package_overrides: dict[str, str],
    visited: set[tuple[str, str]],
):
    names = set(names or set())
    if module_name == "utils":
        names.discard("get_executable_dir")
        if not names:
            return

    if module_name == "core.model_assets" and ("MODEL_IGNORE_PATTERNS" in names or not names):
        from core.model_assets import MODEL_IGNORE_PATTERNS
        chunk_key = ("core.model_assets", "MODEL_IGNORE_PATTERNS")
        if chunk_key not in visited:
            support_chunks.append(f"MODEL_IGNORE_PATTERNS = {repr(tuple(MODEL_IGNORE_PATTERNS))}")
            visited.add(chunk_key)
        names.discard("MODEL_IGNORE_PATTERNS")
        if not names:
            return

    rel_path = _module_name_to_rel_path(module_name)
    if rel_path is None:
        return
    source = _read_project_source(rel_path)
    if not source.strip():
        return

    _add_import_context_from_source(source, import_lines, required_imports, package_overrides)
    for model_file in _collect_model_file_literals(source):
        if model_file not in required_model_files:
            required_model_files.append(model_file)

    definitions, assignments, imported_symbols = _collect_module_symbols(source)
    if not names or "*" in names:
        chunk_key = (module_name, "*")
        if chunk_key not in visited:
            support_chunks.append(
                f"# ---- Inlined dependency: {rel_path} ----\n"
                + _strip_local_project_imports(source).strip()
            )
            visited.add(chunk_key)
        return

    for name in sorted(names):
        chunk_key = (module_name, name)
        if chunk_key in visited:
            continue
        if name in definitions:
            segment = _safe_source_segment(source, definitions[name]).strip()
        elif name in assignments:
            segment = _safe_source_segment(source, assignments[name]).strip()
        else:
            continue
        if not segment:
            continue
        support_chunks.append(f"# ---- Inlined dependency: {module_name}.{name} ----\n{segment}")
        visited.add(chunk_key)
        _resolve_source_dependencies(
            segment,
            source,
            imported_symbols,
            definitions,
            assignments,
            module_name,
            support_chunks,
            import_lines,
            required_imports,
            required_model_files,
            package_overrides,
            visited,
        )

def _resolve_source_dependencies(
    source: str,
    module_source: str,
    imported_symbols: dict[str, tuple[str | None, str | None, str]],
    definitions: dict[str, ast.AST],
    assignments: dict[str, ast.AST],
    current_module: str,
    support_chunks: list[str],
    import_lines: list[str],
    required_imports: dict[str, str],
    required_model_files: list[str],
    package_overrides: dict[str, str],
    visited: set[tuple[str, str]],
):
    _add_import_context_from_source(source, import_lines, required_imports, package_overrides)
    for model_file in _collect_model_file_literals(source):
        if model_file not in required_model_files:
            required_model_files.append(model_file)

    for name in sorted(_collect_global_load_names(source)):
        if name == "get_executable_dir":
            continue
        if name in definitions:
            chunk_key = (current_module, name)
            if chunk_key not in visited:
                source_for_segment = module_source
                if not source_for_segment:
                    module_rel = _module_name_to_rel_path(current_module)
                    source_for_segment = _read_project_source(module_rel) if module_rel else ""
                if source_for_segment:
                    segment = _safe_source_segment(source_for_segment, definitions[name]).strip()
                    if segment:
                        support_chunks.append(f"# ---- Inlined dependency: {current_module}.{name} ----\n{segment}")
                        visited.add(chunk_key)
                        _resolve_source_dependencies(
                            segment,
                            source_for_segment,
                            imported_symbols,
                            definitions,
                            assignments,
                            current_module,
                            support_chunks,
                            import_lines,
                            required_imports,
                            required_model_files,
                            package_overrides,
                            visited,
                        )
            continue
        if name in assignments:
            chunk_key = (current_module, name)
            if chunk_key not in visited:
                source_for_segment = module_source
                if not source_for_segment:
                    module_rel = _module_name_to_rel_path(current_module)
                    source_for_segment = _read_project_source(module_rel) if module_rel else ""
                if source_for_segment:
                    segment = _safe_source_segment(source_for_segment, assignments[name]).strip()
                    if segment:
                        support_chunks.append(f"# ---- Inlined dependency: {current_module}.{name} ----\n{segment}")
                        visited.add(chunk_key)
                        _resolve_source_dependencies(
                            segment,
                            source_for_segment,
                            imported_symbols,
                            definitions,
                            assignments,
                            current_module,
                            support_chunks,
                            import_lines,
                            required_imports,
                            required_model_files,
                            package_overrides,
                            visited,
                        )
            continue
        if name not in imported_symbols:
            continue
        module_name, imported_name, line = imported_symbols[name]
        if module_name and _is_local_project_module(module_name):
            _inline_local_module_symbols(
                module_name,
                {imported_name or name},
                support_chunks,
                import_lines,
                required_imports,
                required_model_files,
                package_overrides,
                visited,
            )
        elif module_name:
            _add_required_import(required_imports, module_name, package_overrides)
            if line and line not in import_lines:
                import_lines.append(line)

def _collect_export_support_context(
    used_functions: dict[str, str],
    all_step_code: str,
    used_tool_names: list[str],
) -> tuple[str, list[str], dict[str, str], list[str], dict[str, dict]]:
    combined_code = "\n\n".join([all_step_code, *used_functions.values()])
    support_chunks: list[str] = []
    import_lines: list[str] = []
    required_imports: dict[str, str] = {
        "cv2": _EXPORT_IMPORT_PACKAGES["cv2"],
        "numpy": _EXPORT_IMPORT_PACKAGES["numpy"],
    }
    required_model_files: list[str] = []
    exported_tool_schemas: dict[str, dict] = {}
    package_overrides: dict[str, str] = {}

    for tool_name in used_tool_names:
        tool = global_registry.tools.get(tool_name)
        schema = tool.get("schema", {}) if isinstance(tool, dict) else {}
        exported_tool_schemas[tool_name] = _export_schema_subset(schema)
        package_overrides.update(_normalize_additional_package_map(schema.get("additional_packages", [])))
        for line in _schema_additional_import_lines(schema):
            if line not in import_lines:
                import_lines.append(line)
        for model_file in _schema_model_files(schema):
            if model_file not in required_model_files:
                required_model_files.append(model_file)

    _add_import_context_from_source(combined_code, import_lines, required_imports, package_overrides)
    for line in import_lines:
        if line.startswith("import "):
            for token in line.removeprefix("import ").split(","):
                _add_required_import(required_imports, token.strip().split(" as ", maxsplit=1)[0], package_overrides)
        elif line.startswith("from "):
            module_name = line.removeprefix("from ").split(" import ", maxsplit=1)[0]
            _add_required_import(required_imports, module_name, package_overrides)
    for model_file in _collect_model_file_literals(combined_code):
        if model_file not in required_model_files:
            required_model_files.append(model_file)

    for tool_name in used_tool_names:
        tool = global_registry.tools.get(tool_name)
        if not isinstance(tool, dict):
            continue
        func = tool.get("raw_func") or tool.get("func")
        if func is None:
            continue
        module = inspect.getmodule(func)
        module_name = getattr(module, "__name__", "")
        try:
            module_source = inspect.getsource(module) if module is not None else ""
        except Exception:
            module_source = ""
        if not module_source:
            module_file = inspect.getsourcefile(func)
            if module_file:
                try:
                    module_source = Path(module_file).read_text(encoding="utf-8")
                except Exception:
                    module_source = ""
        definitions, assignments, imported_symbols = _collect_module_symbols(module_source)
        _resolve_source_dependencies(
            used_functions.get(tool_name, ""),
            module_source,
            imported_symbols,
            definitions,
            assignments,
            module_name,
            support_chunks,
            import_lines,
            required_imports,
            required_model_files,
            package_overrides,
            set(),
        )

    support_code = "\n\n".join(support_chunks)
    _add_import_context_from_source(support_code, import_lines, required_imports, package_overrides)
    for model_file in _collect_model_file_literals(support_code):
        if model_file not in required_model_files:
            required_model_files.append(model_file)
    return support_code, required_model_files, required_imports, import_lines, exported_tool_schemas

def _format_export_dependency_block(required_imports: dict[str, str]) -> str:
    package_map = dict(sorted(required_imports.items()))
    return f"""REQUIRED_IMPORTS = {repr(package_map)}

def _check_required_imports():
    missing = []
    for import_name, package_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    if missing:
        unique_missing = []
        for package_name in missing:
            if package_name not in unique_missing:
                unique_missing.append(package_name)
        install_cmd = sys.executable + " -m pip install " + " ".join(unique_missing)
        raise SystemExit(
            "缺少运行依赖，导出的脚本无法继续执行。\\n"
            "请先运行：\\n"
            f"  {{install_cmd}}"
        )

_check_required_imports()
"""

def _format_export_import_lines(import_lines: list[str], required_imports: dict[str, str]) -> str:
    lines = ["import cv2", "import numpy as np"]
    default_imports = {
        "PIL": ["import PIL", "from PIL import Image, ImageFilter"],
        "skimage": ["import skimage", "from skimage import *", "from skimage import util, restoration, exposure"],
        "scipy": ["import scipy"],
        "torch": ["import torch", "import torch.nn as nn", "import torch.nn.functional as F"],
        "torchvision": ["import torchvision", "import torchvision.transforms.functional as TF"],
        "modelscope": [
            "from modelscope.pipelines import pipeline",
            "from modelscope.utils.constant import Tasks",
            "from modelscope.outputs import OutputKeys",
        ],
    }
    for import_name in sorted(required_imports):
        for line in default_imports.get(import_name, [f"import {import_name}"]):
            if line not in lines:
                lines.append(line)
    for line in import_lines:
        stripped = line.strip()
        if stripped and stripped not in lines:
            lines.append(stripped)
    return "\n".join(lines)

def _format_export_runtime_helpers(
    required_model_files: list[str],
    exported_tool_schemas: dict[str, dict],
) -> str:
    return f"""def get_executable_dir() -> Path:
    return Path(__file__).resolve().parent

os.environ.setdefault("TORCH_HOME", str(get_executable_dir() / "caches/model_assets/torch"))
os.environ.setdefault("HF_HOME", str(get_executable_dir() / "caches/model_assets/huggingface"))
os.environ.setdefault("MODELSCOPE_CACHE", str(get_executable_dir() / "caches/model_assets/modelscope"))

REQUIRED_MODEL_FILES = {repr(required_model_files)}
EXPORTED_TOOL_SCHEMAS = {repr(exported_tool_schemas)}

def _safe_repo_cache_name(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(repo_id or "").strip())

def resolve_export_model_dir(schema: dict | None) -> str:
    if not isinstance(schema, dict):
        return ""
    source = str(schema.get("model_source") or schema.get("source") or "").strip().lower()
    repo_id = str(schema.get("model_repo_id") or schema.get("repo_id") or "").strip()
    if source in {{"github", "huggingface", "modelscope"}}:
        base = get_executable_dir() / "caches" / "model_assets" / source
        return str((base / _safe_repo_cache_name(repo_id)).resolve()) if repo_id else str(base.resolve())
    for key in ("model_dir", "download_dir", "model_download_dir", "model_asset_dir"):
        value = schema.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value.strip())
        return str(path if path.is_absolute() else (get_executable_dir() / path).resolve())
    return ""

def _check_required_model_files():
    missing = [
        rel_path
        for rel_path in REQUIRED_MODEL_FILES
        if not (get_executable_dir() / rel_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少模型权重文件。请把项目的 models 目录放到导出脚本同级，"
            "或修改 get_executable_dir() 指向包含这些文件的目录：\\n"
            + "\\n".join(f"- {{path}}" for path in missing)
        )
"""

def build_export_script_for_message(messages: list[dict], current_index: int) -> str | None:
    chain = _collect_export_process_chain(messages, current_index)
    if not chain:
        return None

    step_codes: list[str] = []
    params_sequence: list[dict] = []
    step_meta: list[dict] = []
    for step_idx, (msg_idx, step_msg) in enumerate(chain, start=1):
        function_name = f"process_step_{step_idx}"
        step_code = _process_code_to_export_step(step_msg.get("process_code", ""), function_name)
        step_codes.append(step_code)
        params_sequence.append(step_msg.get("best_params", {}))
        step_meta.append({
            "function": function_name,
            "message_index": msg_idx,
            "input_source": step_msg.get("input_source", "original"),
        })

    all_step_code = "\n\n".join(step_codes)
    matches = _extract_used_operator_names(all_step_code)

    if matches:
        logger.info(f"检测到的函数调用: {set(matches)}")
    else:
        logger.info("未检测到任何 cv_wrappers 调用")

    used_functions = extract_funcs(matches)
    logger.info(f"最终 used_functions 包含的函数: {list(used_functions.keys())}")

    wrapper_class_code = ""
    if used_functions:
        wrapper_methods = "\n\n".join(used_functions.values())
        wrapper_class_code = f"""
class cv_wrappers:
{wrapper_methods}

"""

    support_code, required_model_files, required_imports, import_lines, exported_tool_schemas = _collect_export_support_context(
        used_functions,
        all_step_code,
        matches,
    )
    dependency_block = _format_export_dependency_block(required_imports)
    import_lines = _format_export_import_lines(import_lines, required_imports)
    runtime_helpers = _format_export_runtime_helpers(required_model_files, exported_tool_schemas)
    tool_alias_code = "\n".join(
        f"{tool_name} = cv_wrappers.{tool_name}"
        for tool_name in used_functions
    )

    step_calls = "\n".join(
        f"    out = process_step_{idx}(out, params_sequence[{idx - 1}], cache)"
        for idx in range(1, len(step_codes) + 1)
    )

    return f"""# -*- coding: utf-8 -*-
# Auto-generated Image Enhancement Script
# Generated by ChatImageEnhance
# Usage: python image_enhancement_script.py <input_dir> <output_dir>

import os
import argparse
import sys
import re
import importlib.util
from pathlib import Path

# 设置标准输出编码为 UTF-8（避免 Windows 命令行乱码）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

{dependency_block}
{import_lines}

{runtime_helpers}

{support_code}

{wrapper_class_code}
{tool_alias_code}

{all_step_code}

# Exported processing chain. Earlier entries are applied first.
process_step_metadata = {repr(step_meta)}
best_params_sequence = {repr(params_sequence)}
best_params = best_params_sequence[-1] if best_params_sequence else {{}}

def process(img, params_sequence=None):
    \"\"\"Apply exported enhancement steps in the same order used in ChatImageEnhance.\"\"\"
    _check_required_model_files()
    if params_sequence is None:
        params_sequence = best_params_sequence
    cache = {{}}
    out = img
{step_calls}
    return out

def batch_process(input_dir, output_dir):
    \"\"\"批量处理图像\"\"\"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"已自动创建输入文件夹：{{input_dir}}")
        print("请将图片放入该文件夹后重新运行脚本")
        return

    os.makedirs(output_dir, exist_ok=True)
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    if not files:
        print(f'未在 "{{input_dir}}" 中找到支持的图片')
        return

    print(f"找到 {{len(files)}} 张图片，开始处理...")
    for i, filename in enumerate(files):
        img_path = os.path.join(input_dir, filename)
        with open(img_path, mode='rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            try:
                enhanced = process(image)
                output_path = os.path.join(output_dir, filename)
                succ, enc_img = cv2.imencode('.png', enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 2])
                with open(output_path, mode='wb') as f:
                    f.write(enc_img.tobytes())
                print(f"[{{i+1}}/{{len(files)}}] 已处理: {{filename}}")
            except Exception as e:
                print(f"[{{i+1}}/{{len(files)}}] 处理失败 {{filename}}: {{str(e)}}")
        else:
            print(f"[{{i+1}}/{{len(files)}}] 读取失败: {{filename}}")

    print(f"\\n处理完成！结果保存至: {{output_dir}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量图像增强脚本')
    parser.add_argument('input_dir', nargs='?', default='input_images', help='输入文件夹')
    parser.add_argument('output_dir', nargs='?', default='output_images', help='输出文件夹')
    parser.add_argument('--dry-run', action='store_true', help='仅预览文件，不处理')

    args = parser.parse_args()

    if args.dry_run:
        print(f"预览模式：将处理 '{{args.input_dir}}' -> '{{args.output_dir}}'")
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        if os.path.exists(args.input_dir):
            for f in os.listdir(args.input_dir):
                if f.lower().endswith(supported_formats):
                    print(f"- {{f}}")
        else:
            print(f"输入文件夹不存在：{{args.input_dir}}")
    else:
        batch_process(args.input_dir, args.output_dir)
"""

def _inject_export_runtime_defaults(source: str, tool_name: str, func: Callable) -> str:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return source

    injections: list[str] = []
    if "model_dir" in sig.parameters:
        injections.extend([
            f'        __schema = EXPORTED_TOOL_SCHEMAS.get("{tool_name}", {{}})',
            '        if model_dir is None or (isinstance(model_dir, str) and not model_dir.strip()):',
            '            model_dir = resolve_export_model_dir(__schema)',
        ])

    if not injections:
        return source

    lines = source.splitlines()
    def_idx = None
    balance = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if def_idx is None:
            if stripped.startswith("def "):
                def_idx = idx
                balance = line.count("(") - line.count(")")
                if balance <= 0 and stripped.endswith(":"):
                    insert_idx = idx + 1
                    break
            continue
        balance += line.count("(") - line.count(")")
        if balance <= 0 and stripped.endswith(":"):
            insert_idx = idx + 1
            break
    else:
        return source

    return "\n".join(lines[:insert_idx] + injections + lines[insert_idx:])

@st.cache_resource()
def extract_funcs(matches: list[str]):
    used_functions = dict()
    for called_func_name in set(matches):  # 去重
        logger.info(f"正在处理函数: {called_func_name}")
        
        # 尝试通过注册表映射查找实际函数
        if called_func_name in global_registry.tools:
            tool = global_registry.tools[called_func_name]
            func: Callable = tool.get('raw_func') or tool['func']
            actual_name = func.__name__
            logger.info(f"-> 映射找到: {called_func_name} -> {func.__name__}")
            
            # 根据模块名获取对应的模块对象
            source = _extract_wrapper_source(func)
            logger.info(f"-> 提取源码: {'成功' if source else '失败'}")
            
            if source:
                # 关键修复：将函数名替换为 LLM 使用的注册表名称
                # 例如：将 "def safe_enhance_clahe(...)" 替换为 "def CLAHE_Enhancement(...)"
                source = re.sub(rf'def\s+{actual_name}\s*\(', f'def {called_func_name}(', source)
                source = _inject_export_runtime_defaults(source, called_func_name, func)
                used_functions[called_func_name] = source
                logger.info(f"-> ✅ 成功添加到 used_functions")
            else:
                logger.info(f"-> ❌ 源码提取失败，跳过")
        else:
            logger.info(f"-> 不在注册表映射中")

    return used_functions
