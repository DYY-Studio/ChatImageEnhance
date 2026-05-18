import streamlit as st
import yaml
import numpy as np
import inspect
import re
import ast
import logging

from tools import global_registry
from components.image_comparison import image_comparison
from utils import get_encoded_img, get_thumbnail_img, get_thumbnail_size, get_executable_dir

from typing import Literal, Callable

logger = logging.getLogger("BaseComponents")

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

def _build_custom_tool_import_lines(tool_code: str, extra_imports: list[str] | None = None) -> list[str]:
    # 基础依赖（与历史行为兼容）
    imports = [
        "import numpy as np",
        "import cv2",
        "import math",
        "import skimage",
        "import scipy",
    ]

    detected = set()
    code_text = str(tool_code or "")
    detection_map = {
        "torch": r"\btorch\b",
        "torchvision": r"\btorchvision\b",
        "transformers": r"\btransformers\b",
        "diffusers": r"\bdiffusers\b",
        "modelscope": r"\bmodelscope\b",
        "huggingface_hub": r"\bhuggingface_hub\b",
        "PIL": r"\bPIL\b",
    }
    for mod_name, pattern in detection_map.items():
        if re.search(pattern, code_text):
            detected.add(mod_name)

    for imp in _normalize_import_list(extra_imports):
        root = imp.split(".", maxsplit=1)[0].strip()
        if root:
            detected.add(imp.strip())

    existing_roots = {
        line.replace("import ", "").replace(" as np", "").strip().split(".", maxsplit=1)[0]
        for line in imports
    }
    for mod_name in sorted(detected):
        root = mod_name.split(".", maxsplit=1)[0]
        if root in existing_roots:
            continue
        imports.append(f"import {mod_name}")
        existing_roots.add(root)

    return imports

def save_tool_to_local(tool: dict, overwrite: bool = True, status_key: str | None = None):
    if not isinstance(tool, dict):
        raise ValueError("工具对象无效")
    if not isinstance(tool.get("schema"), dict):
        raise ValueError("工具 schema 无效")

    tool_schema = dict(tool['schema'])
    tool_code = str(tool.get('code') or "")
    if not tool_code.strip():
        raise ValueError("工具代码为空")

    tool_name = global_registry.sanitize_tool_name(tool_schema.get('name', ''))
    _, custom_tool_dir, file_base = global_registry.get_custom_tool_paths(tool_name)
    custom_tool_dir.mkdir(parents=True, exist_ok=True)

    tool_yaml = file_base.with_suffix(".yaml")
    tool_py = file_base.with_suffix(".py")
    if (not overwrite) and (tool_yaml.exists() or tool_py.exists()):
        raise FileExistsError(f"工具 {tool_name} 已存在")

    extra_imports = _normalize_import_list(tool.get("additional_imports"))
    if not extra_imports:
        extra_imports = _normalize_import_list(tool_schema.get("additional_imports"))

    extra_packages = _normalize_import_list(tool.get("additional_packages"))
    if not extra_packages:
        extra_packages = _normalize_import_list(tool_schema.get("additional_packages"))
    for package in extra_packages:
        import_name = _package_to_import_name(package)
        if import_name and import_name not in extra_imports:
            extra_imports.append(import_name)

    # 将附加导入信息一并持久化，便于后续审计与二次编辑
    if extra_imports:
        tool_schema["additional_imports"] = extra_imports
    if extra_packages:
        tool_schema["additional_packages"] = extra_packages

    tool_yaml.write_text(
        yaml.dump(tool_schema, allow_unicode=True, indent=2),
        encoding='utf-8'
    )

    imports_text = "\n".join(_build_custom_tool_import_lines(tool_code, extra_imports))
    tool_py.write_text(
        f"{imports_text}\n\n{tool_code}",
        encoding='utf-8'
    )

    succ, err = global_registry.load_custom_tool(tool_name)
    if status_key:
        st.session_state[status_key] = {
            "ok": bool(succ),
            "message": (
                f"工具 {tool_name} 已保存并加载成功"
                if succ else
                f"工具 {tool_name} 已保存，但加载失败：{err}"
            )
        }

def render_tool_save_button(
    tool: dict,
    button_label: str = "🆕 保存新工具",
    button_key: str | None = None,
    allow_overwrite: bool = True
):
    status_key = f"{button_key}_status" if button_key else None
    if status_key and status_key in st.session_state:
        status = st.session_state.pop(status_key)
        if status.get("ok"):
            st.success(status.get("message", "工具已保存"))
        else:
            st.warning(status.get("message", "工具保存后加载失败"))

    if not isinstance(tool, dict):
        st.button(button_label, disabled=True, key=button_key)
        return
    if not isinstance(tool.get("schema"), dict):
        st.button(button_label, disabled=True, key=button_key)
        return

    raw_tool_name = tool["schema"].get("name")
    if not raw_tool_name:
        st.button(button_label, disabled=True, key=button_key)
        return
    try:
        tool_name = global_registry.sanitize_tool_name(raw_tool_name)
        _, _, file_name = global_registry.get_custom_tool_paths(tool_name)
    except Exception as e:
        st.button(button_label, disabled=True, key=button_key)
        st.warning(f"工具名不合法，已阻止保存：{e}")
        return

    has_local_tool = file_name.with_suffix(".yaml").exists() and file_name.with_suffix(".py").exists()

    if has_local_tool and allow_overwrite:
        st.button(
            "♻️ 覆盖保存工具",
            on_click=save_tool_to_local,
            args=[tool, True, status_key],
            key=button_key
        )
        if tool_name not in global_registry.tools:
            succ, err = global_registry.load_custom_tool(tool_name)
            if not succ and err:
                st.warning(f"本地工具加载失败：{tool_name}，错误：{err}")
    elif has_local_tool:
        st.button(button_label, disabled=True, key=button_key)
        if tool_name not in global_registry.tools:
            succ, err = global_registry.load_custom_tool(tool_name)
            if not succ and err:
                st.warning(f"本地工具加载失败：{tool_name}，错误：{err}")
    else:
        global_registry.dynamic_unregister(tool_name)
        st.button(
            button_label,
            on_click=save_tool_to_local,
            args=[tool, False, status_key],
            key=button_key
        )

    tool_load_error = global_registry.last_custom_tool_errors.get(tool_name)
    if tool_load_error:
        st.warning(f"本地工具存在加载异常：{tool_name}，错误：{tool_load_error}")

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

def get_thumbnail_img_wrapper(
    raw_array: np.ndarray, 
    mode: Literal["binary", "b64", "array"]
) -> bytes | str | np.ndarray | None:
    return get_thumbnail_img(
        raw_array, mode, st.session_state['preview_img_max_side'], st.session_state['preview_img_scale']
    )

def get_previous_img(curr_idx: int, ignore_test_mode: bool = True):
    prev_image = None
    if curr_idx > 0:
        # 向前查找最近一个包含图像的assistant消息
        for i in range(curr_idx - 1, -1, -1):
            prev_msg = st.session_state.messages[i]
            if prev_msg["role"] == "assistant" and "image" in prev_msg:
                if ignore_test_mode and "test_mode" in prev_msg:
                    continue
                prev_image = prev_msg["image"]
                break
    return prev_image

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

def _next_user_feedback(messages: list[dict], assistant_index: int, stop_index: int) -> str:
    for msg in messages[assistant_index + 1:stop_index]:
        if msg.get("role") == "user":
            return str(msg.get("content") or "").strip()
    return ""

def _build_attempt_history_summary(
    messages: list[dict],
    last_assistant_index: int,
    max_entries: int = 6
) -> str:
    if last_assistant_index <= 0:
        return ""

    entries: list[str] = []
    round_no = 0
    for idx, msg in enumerate(messages[:last_assistant_index]):
        if msg.get("role") != "assistant" or "image" not in msg or msg.get("test_mode"):
            continue
        round_no += 1
        operators = _extract_used_operator_names(msg.get("process_code", ""))
        feedback = _next_user_feedback(messages, idx, last_assistant_index)
        operators_text = ", ".join(operators) if operators else "未检测到明确算子"
        feedback_text = feedback if feedback else "未记录后续用户反馈"
        entries.append(
            f"{round_no}. 使用算子: {operators_text}\n"
            f"   后续用户反馈: {feedback_text}"
        )

    if max_entries > 0:
        entries = entries[-max_entries:]
    return "\n".join(entries)

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

    def replace_trial_suggest(match):
        param_name = match.group(1)
        return f'params["{param_name}"]'

    return re.sub(
        r"trial\.suggest_[a-zA-Z_]+\([\"']([a-zA-Z0-9_]+)[\"'][^)]*\)",
        replace_trial_suggest,
        final_func_code,
    )

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

    used_functions = {}
    all_step_code = "\n\n".join(step_codes)
    matches = re.findall(r"(?:cv_wrappers)\.(\w+)\s*\(", all_step_code)

    if matches:
        logger.info(f"检测到的函数调用: {set(matches)}")
    else:
        logger.info("未检测到任何 cv_wrappers/skimage_wrappers 调用")

    extract_funcs(matches, used_functions)
    logger.info(f"最终 used_functions 包含的函数: {list(used_functions.keys())}")

    wrapper_class_code = ""
    if used_functions:
        wrapper_methods = "\n\n".join(used_functions.values())
        wrapper_class_code = f"""
class cv_wrappers:
{wrapper_methods}

"""

    step_calls = "\n".join(
        f"    out = process_step_{idx}(out, params_sequence[{idx - 1}], cache)"
        for idx in range(1, len(step_codes) + 1)
    )

    return f"""# -*- coding: utf-8 -*-
# Auto-generated Image Enhancement Script
# Generated by ChatImageEnhance
# Usage: python image_enhancement_script.py <input_dir> <output_dir>

import cv2
import skimage
import numpy as np
import PIL
from skimage import *
import os
import argparse
import sys

# 设置标准输出编码为 UTF-8（避免 Windows 命令行乱码）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

{wrapper_class_code}
{all_step_code}

# Exported processing chain. Earlier entries are applied first.
process_step_metadata = {repr(step_meta)}
best_params_sequence = {repr(params_sequence)}
best_params = best_params_sequence[-1] if best_params_sequence else {{}}

def process(img, params_sequence=None):
    \"\"\"Apply exported enhancement steps in the same order used in ChatImageEnhance.\"\"\"
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

def delete_message(idx: int, target_only: bool = False):
    msgs: list = st.session_state.messages
    clear_encoded_cache = False
    if not target_only:
        target_msg = msgs[idx]
        if target_msg['role'] == "user":
            if len(msgs) > idx + 1 and msgs[idx + 1]['role'] == "assistant":
                if len(msgs) == idx + 2 and 'image' in msgs[idx + 1]:
                    st.session_state['best_bgr'] = get_previous_img(idx + 1)
                    clear_encoded_cache = True
                msgs.pop(idx + 1)
            msgs.pop(idx)
        elif target_msg['role'] == "assistant":
            if len(msgs) == idx + 1 and 'image' in target_msg:
                st.session_state['best_bgr'] = get_previous_img(idx)
                clear_encoded_cache = True
            msgs.pop(idx)
            if idx > 0 and msgs[idx - 1]['role'] == "user":
                msgs.pop(idx - 1)
    else:
        msgs.pop(idx)

    if clear_encoded_cache:
        get_encoded_img.clear()

@st.cache_resource()
def extract_funcs(matches: list[str], used_functions: dict):
    for called_func_name in set(matches):  # 去重
        logger.info(f"正在处理函数: {called_func_name}")
        
        # 尝试通过注册表映射查找实际函数
        if called_func_name in global_registry.tools:
            func: Callable = global_registry.tools[called_func_name]['func']
            actual_name = func.__name__
            logger.info(f"-> 映射找到: {called_func_name} -> {func.__name__}")
            
            # 根据模块名获取对应的模块对象
            source = _extract_wrapper_source(func)
            logger.info(f"-> 提取源码: {'成功' if source else '失败'}")
            
            if source:
                # 关键修复：将函数名替换为 LLM 使用的注册表名称
                # 例如：将 "def safe_enhance_clahe(...)" 替换为 "def CLAHE_Enhancement(...)"
                source = re.sub(rf'def\s+{actual_name}\s*\(', f'def {called_func_name}(', source)
                used_functions[called_func_name] = source
                logger.info(f"-> ✅ 成功添加到 used_functions")
            else:
                logger.info(f"-> ❌ 源码提取失败，跳过")
        else:
            logger.info(f"-> 不在注册表映射中")

def render_message_content(msg, index: int):
    """提取内部渲染逻辑，供历史记录与最新消息复用"""
    if msg.get("error"):
        st.error(msg.get("content", "本轮运行失败"))
        details = str(msg.get("error_details") or msg.get("error") or "").strip()
        if details:
            with st.expander("错误详情"):
                st.code(details, language="text")
        if st.button("🚮 删除本轮对话", on_click=delete_message, args=[index], key=f"del_btn_{id(msg)}_{index}"):
            st.rerun()
        return

    st.markdown(msg["content"])
    if "image" not in msg:
        if st.button("🚮 删除本轮对话", on_click=delete_message, args=[index], key=f"del_btn_{id(msg)}_{index}"):
            st.rerun()
    else:
        meta_items = []
        if msg.get("image") is not None:
            try:
                h, w = msg["image"].shape[:2]
                meta_items.append(f"图像 {w}x{h}")
            except Exception:
                pass
        if "n_trials_used" in msg:
            meta_items.append(f"实际调优 {msg.get('n_trials_used')} 轮")
        if msg.get("new_tool"):
            tool_schema = msg["new_tool"].get("schema", {}) if isinstance(msg["new_tool"], dict) else {}
            tool_name = tool_schema.get("name") or "未命名工具"
            meta_items.append(f"新增工具 {tool_name}")
        if meta_items:
            st.caption(" · ".join(meta_items))

        if any(key in msg for key in ['eval_code', 'process_code', 'best_params']):
            with st.expander("🛠️ 查看此轮生成的代码与最优参数"):
                with st.expander("评价逻辑 (Evaluation Code)"):
                    st.code(msg.get("eval_code", "# 无评价代码"), language="python")
                
                with st.expander("处理逻辑 (Process Code)"):
                    st.code(msg.get("process_code", "# 无处理代码"), language="python")
                
                with st.expander("Optuna 最优参数组合"):
                    st.json(msg.get("best_params", {}))
                if msg.get("new_tool"):
                    new_tool = msg["new_tool"]
                    with st.expander("新增工具详情"):
                        st.write("Schema")
                        st.json(new_tool.get("schema", {}))
                        st.write("代码")
                        st.code(new_tool.get("code", "# 无工具代码"), language="python")
                        if new_tool.get("additional_imports"):
                            st.write("附加导入")
                            st.json(new_tool.get("additional_imports"))
                        if new_tool.get("additional_packages"):
                            st.write("附加依赖")
                            st.json(new_tool.get("additional_packages"))
        else:
            st.info(":information_source: 本轮处理没有任何可显示的信息")

        with st.container(horizontal=True):
            if st.button("🚮 删除本轮对话", on_click=delete_message, args=[index], key=f"del_btn_{id(msg)}_{index}"):
                st.rerun()

            succ, enc_img_bytes = get_encoded_img(msg["image"])
            if succ:
                st.download_button(
                    label="📥 保存此版本", 
                    data=enc_img_bytes, 
                    file_name=f"enhanced_history_{index}.png", 
                    mime="image/png", 
                    key=f"dl_history_{index}"
                )
            else:
                st.button("📥 保存此版本", disabled=True)
            
            if "new_tool" in msg and msg["new_tool"]:
                render_tool_save_button(
                    msg["new_tool"],
                    button_label="🆕 保存新工具",
                    button_key=f"save_tool_chat_{index}"
                )

            process_code = msg.get("process_code", "")
            script_content = build_export_script_for_message(st.session_state.messages, index)

            if not process_code or script_content is None:
                st.button("📦 导出为处理脚本", disabled=True, help="需要生成处理代码和最优参数才能导出")
            else:
                st.download_button(
                    label="📦 导出为处理脚本",
                    data=script_content,
                    file_name="image_enhancement_script.py",
                    mime="text/x-python",
                    key=f"export_script_{id(msg)}_{index}"
                )

        prev_image = get_previous_img(index, ignore_test_mode=False)

        with st.container(border=False):
            comp_target = "原图"
            if prev_image is not None:
                # ===== [修改] 添加唯一的 key 以避免 ID 冲突 =====
                comp_target = st.pills(
                    "对比对象", 
                    ["原图", "上一轮"], 
                    default="原图", 
                    required=True,
                    key=f"comp_target_{index}"  # 使用消息索引作为唯一标识
                )

            image_comparison(
                get_thumbnail_img_wrapper(st.session_state['img_bgr'], 'b64') if comp_target == "原图" else get_thumbnail_img_wrapper(prev_image, 'b64'),
                get_thumbnail_img_wrapper(msg["image"], 'b64'),
                get_thumbnail_size(msg["image"], st.session_state['preview_img_max_side'])[1],
                comp_target,
                "最新"
            )

def generate_user_prompt(
    user_feedback: str,
    include_process: bool = False, 
    include_evaluate: bool = False, 
    step_by_step: bool = False
):
    messages = list(st.session_state.messages)
    prior_messages = messages[:-1]
    last_assistant_index = next(
        (
            idx for idx in range(len(prior_messages) - 1, -1, -1)
            if prior_messages[idx].get('role') == 'assistant'
            and "image" in prior_messages[idx]
            and "test_mode" not in prior_messages[idx]
        ),
        None
    )
    last_assistant_msg = (
        prior_messages[last_assistant_index]
        if last_assistant_index is not None else None
    )
    
    current_iter_prompt = f""
    if last_assistant_msg:
        initial_user_msg = next(
            (
                m for m in prior_messages
                if m.get("role") == "user" and str(m.get("content") or "").strip()
            ),
            None
        )
        if initial_user_msg:
            current_iter_prompt += (
                f"--- 初始用户要求 ---\n{initial_user_msg['content']}\n\n"
            )

        history_summary = _build_attempt_history_summary(
            prior_messages,
            last_assistant_index,
        )
        if history_summary:
            current_iter_prompt += (
                "--- 历史尝试摘要（更早轮次，仅包含算子列表和后续用户反馈） ---\n"
                f"{history_summary}\n\n"
            )

        current_iter_prompt += (
            "--- 本轮输入图像来源 ---\n"
            f"{'上一轮结果图像' if step_by_step else '原始图像'}\n\n"
        )

        current_iter_prompt += f"--- 上一轮执行状态/系统回复 ---\n{last_assistant_msg['content']}\n"

        l_params = last_assistant_msg.get("best_params", {})
        l_eval = last_assistant_msg.get("eval_code", "")
        l_proc = last_assistant_msg.get("process_code", "")
        
        if l_eval and l_proc:
            if include_evaluate: current_iter_prompt += f"\n--- 上一轮使用的评价函数代码 ---\n```python\n{l_eval}\n```\n"
            if include_process: current_iter_prompt += f"\n--- 上一轮使用的图像处理代码 ---\n```python\n{l_proc}\n```\n"

        if l_params:
            current_iter_prompt += f"\n--- 上一轮 Optuna 搜索到的最优参数 ---\n{l_params}\n"

        current_iter_prompt += (
            f"\n--- 本轮用户最新反馈/要求 ---\n{user_feedback}\n"
            "\n请结合历史尝试摘要中的算子列表和用户原文反馈，自行判断是否应继续沿用、调整、切换算子，"
            "或在现有算子不足时请求新工具。不要假设历史反馈已经被系统预先判定为正面或负面。"
        )
    else:
        current_iter_prompt += f"--- 用户要求 ---\n{user_feedback}"
    return current_iter_prompt
