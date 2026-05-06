import streamlit as st
import yaml
import numpy as np
import inspect
import re
import ast

from components.image_comparison import image_comparison
from utils import get_encoded_img, get_thumbnail_img, get_thumbnail_size, get_executable_dir

# 导入工具模块（用于动态提取函数源码）
try:
    import tools.cv_wrappers as cv_wrappers_module
    import tools.skimage_wrappers as skimage_wrappers_module
except ImportError as e:
    # 如果导入失败，设置为 None，后续使用时会跳过
    cv_wrappers_module = None
    skimage_wrappers_module = None
    print(f"Warning: Failed to import tool wrappers: {e}")

from typing import Literal

def _extract_wrapper_source(wrapper_module, func_name: str) -> str:
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
        func = getattr(wrapper_module, func_name, None)
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
        print(f"Warning: Failed to extract source for {func_name}: {e}")
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

def render_message_content(msg, index: int):
    """提取内部渲染逻辑，供历史记录与最新消息复用"""
    st.markdown(msg["content"])
    if "image" not in msg:
        if st.button("🚮 删除本轮对话", on_click=delete_message, args=[index], key=f"del_btn_{id(msg)}_{index}"):
            st.rerun()
    else:
        with st.expander("🛠️ 查看此轮生成的代码与最优参数"):
            with st.expander("评价逻辑 (Evaluation Code)"):
                st.code(msg.get("eval_code", "# 无评价代码"), language="python")
            
            with st.expander("处理逻辑 (Process Code)"):
                st.code(msg.get("process_code", "# 无处理代码"), language="python")
            
            with st.expander("Optuna 最优参数组合"):
                st.json(msg.get("best_params", {}))

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
                def save_tool(tool: dict):
                    custom_tool_dir = get_executable_dir() / "tools/custom"
                    custom_tool_dir.mkdir(parents=True, exist_ok=True)

                    tool_schema = tool['schema']
                    tool_code = tool['code']

                    tool_name: str = tool_schema['name']

                    (custom_tool_dir / tool_name).with_suffix(".yaml").write_text(
                        yaml.dump(tool_schema, allow_unicode=True, indent=2), encoding='utf-8'
                    )

                    imports_text = "import numpy as np\nimport cv2\nimport math\nimport skimage\nimport scipy"

                    (custom_tool_dir / tool_name).with_suffix(".py").write_text(
                        f"{imports_text}\n\n{tool_code}", encoding='utf-8'
                    )

                file_name = get_executable_dir() / f"tools/custom/{msg['new_tool']['schema']['name']}"

                if file_name.with_suffix(".yaml").exists() and file_name.with_suffix(".py").exists():
                    st.button("🆕 保存新工具", disabled=True)
                else:
                    st.button("🆕 保存新工具", on_click=save_tool, args=[msg['new_tool']])

            process_code = msg.get("process_code", "")
            best_params = msg.get("best_params", {})
            if process_code and best_params:
                # 移除 Markdown 
                cleaned_code = re.sub(r'^```python\n|```$', '', process_code, flags=re.MULTILINE).strip()
                
                # 提取并标准化 process 函数
                try:
                    tree = ast.parse(cleaned_code)
                    func_node = None
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and node.name == 'process':
                            func_node = node
                            break
                    
                    if func_node:
                        # 获取函数体原始代码片段
                        start_line = func_node.body[0].lineno - 1
                        end_line = func_node.end_lineno
                        lines = cleaned_code.split('\n')[start_line:end_line]
                        
                        # 计算基准缩进并重置
                        min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
                        normalized_lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
                        
                        # 重新组合函数定义
                        final_func_code = "def process(img, params):\n" + "\n".join(["    " + line for line in normalized_lines])
                    else:
                        final_func_code = cleaned_code
                except Exception:
                    final_func_code = cleaned_code

                #移除 Optuna 
                final_func_code = re.sub(r'\s*trial\s*=\s*.*?\n', '\n', final_func_code)  # 移除 trial 定义
                
                def replace_trial_suggest(match):
                    param_name = match.group(1)
                    return f'params["{param_name}"]'
                
                # 匹配所有 trial.suggest_* 调用，包括其所有参数
                final_func_code = re.sub(r'trial\.suggest_[a-zA-Z_]+\(["\']([a-zA-Z0-9_]+)["\'][^)]*\)', replace_trial_suggest, final_func_code)
            
                # 3.2 检测并提取实际使用的自定义工具函数
                # 建立注册表名称到实际函数名的映射（根据 tools/__init__.py 中的注册信息）
                registry_to_actual = {
                    # cv_wrappers 映射
                    'Bilateral_Filter': ('cv_wrappers', 'safe_denoise_bilateral'),
                    'CLAHE_Enhancement': ('cv_wrappers', 'safe_enhance_clahe'),
                    'Gamma_Correction': ('cv_wrappers', 'safe_gamma_correction'),
                    'Unsharp_Masking': ('cv_wrappers', 'safe_unsharp_masking'),
                    'Laplacian_Sharpening': ('cv_wrappers', 'safe_laplacian_sharpening'),
                    'Kernel_Sharpening': ('cv_wrappers', 'safe_kernel_sharpening'),
                    'Auto_Canny': ('cv_wrappers', 'safe_auto_canny'),
                    'Gaussian_Blur': ('cv_wrappers', 'safe_gaussian_blur'),
                    'Morphology_Cleanup': ('cv_wrappers', 'safe_morphology_transform'),
                    'Adaptive_Binarization': ('cv_wrappers', 'safe_adaptive_threshold'),
                    'Median_Denoise': ('cv_wrappers', 'safe_median_blur'),
                    'Auto_White_Balance': ('cv_wrappers', 'safe_color_balance'),
                    'Guided_Filter': ('cv_wrappers', 'safe_guided_filter'),
                    'Image_Deringing': ('cv_wrappers', 'safe_deringing'),
                    'Saturation_Boost_Nonlinear': ('cv_wrappers', 'safe_hsv_saturation_nonlinear'),
                    'Vibrance': ('cv_wrappers', 'safe_vibrance'),
                    'Color_Temperature_Tune': ('cv_wrappers', 'safe_color_temperature'),
                    'Global_Hue_Shift': ('cv_wrappers', 'safe_hue_shift'),
                    'NL_Means_Denoising': ('cv_wrappers', 'safe_nl_means_denoise'),
                    'Sauvola_Binarization': ('cv_wrappers', 'safe_enhance_sauvola'),
                    
                    # skimage_wrappers 映射（如果有）
                    # 可以根据需要添加更多映射
                }
                
                # 从 LLM 生成的代码中提取所有 cv_wrappers.xxx 或 skimage_wrappers.xxx 调用
                used_functions = {}
                
                # 匹配所有 cv_wrappers.func_name(...) 或 skimage_wrappers.func_name(...) 调用
                matches = re.findall(r'(?:cv_wrappers|skimage_wrappers)\.(\w+)\s*\(', final_func_code)
                
                # 调试信息
                if matches:
                    print(f"[DEBUG] 检测到的函数调用: {set(matches)}")
                    print(f"[DEBUG] cv_wrappers 模块状态: {'已加载' if cv_wrappers_module is not None else '未加载'}")
                    print(f"[DEBUG] skimage_wrappers 模块状态: {'已加载' if skimage_wrappers_module is not None else '未加载'}")
                else:
                    print(f"[DEBUG] 未检测到任何 cv_wrappers/skimage_wrappers 调用")
                
                for called_func_name in set(matches):  # 去重
                    print(f"[DEBUG] 正在处理函数: {called_func_name}")
                    
                    # 尝试通过注册表映射查找实际函数
                    if called_func_name in registry_to_actual:
                        module_name, actual_name = registry_to_actual[called_func_name]
                        print(f"[DEBUG]   -> 映射找到: {called_func_name} -> {module_name}.{actual_name}")
                        
                        # 根据模块名获取对应的模块对象
                        if module_name == 'cv_wrappers' and cv_wrappers_module is not None:
                            source = _extract_wrapper_source(cv_wrappers_module, actual_name)
                            print(f"[DEBUG]   -> 从 cv_wrappers 提取源码: {'成功' if source else '失败'}")
                        elif module_name == 'skimage_wrappers' and skimage_wrappers_module is not None:
                            source = _extract_wrapper_source(skimage_wrappers_module, actual_name)
                            print(f"[DEBUG]   -> 从 skimage_wrappers 提取源码: {'成功' if source else '失败'}")
                        else:
                            source = None
                            print(f"[DEBUG]   -> 模块不可用: module_name={module_name}, cv_wrappers={'None' if cv_wrappers_module is None else 'OK'}, skimage_wrappers={'None' if skimage_wrappers_module is None else 'OK'}")
                        
                        if source:
                            # 关键修复：将函数名替换为 LLM 使用的注册表名称
                            # 例如：将 "def safe_enhance_clahe(...)" 替换为 "def CLAHE_Enhancement(...)"
                            source = re.sub(rf'def\s+{actual_name}\s*\(', f'def {called_func_name}(', source)
                            used_functions[called_func_name] = source
                            print(f"[DEBUG]   -> ✅ 成功添加到 used_functions")
                        else:
                            print(f"[DEBUG]   -> ❌ 源码提取失败，跳过")
                    else:
                        print(f"[DEBUG]   -> 不在注册表映射中，尝试直接匹配")
                        # 如果不在映射表中，尝试直接匹配实际函数名（处理特殊情况）
                        if cv_wrappers_module is not None and hasattr(cv_wrappers_module, called_func_name):
                            source = _extract_wrapper_source(cv_wrappers_module, called_func_name)
                            if source:
                                used_functions[called_func_name] = source
                                print(f"[DEBUG]   -> ✅ 从 cv_wrappers 直接匹配成功")
                        elif skimage_wrappers_module is not None and hasattr(skimage_wrappers_module, called_func_name):
                            source = _extract_wrapper_source(skimage_wrappers_module, called_func_name)
                            if source:
                                used_functions[called_func_name] = source
                                print(f"[DEBUG]   -> ✅ 从 skimage_wrappers 直接匹配成功")
                        else:
                            print(f"[DEBUG]   -> ❌ 直接匹配也失败")
                
                print(f"[DEBUG] 最终 used_functions 包含的函数: {list(used_functions.keys())}")
                
                # 构建只包含已使用函数的工具类
                wrapper_class_code = ""
                if used_functions:
                    wrapper_methods = '\n\n'.join(used_functions.values())
                    wrapper_class_code = f"""
# ===================== 内联自定义工具封装（仅包含实际使用的函数）=====================
class cv_wrappers:
{wrapper_methods}

# ============================================================================================
"""

                # 通用：移除 from tools. 开头的导入
                final_func_code = re.sub(r'^from tools\..*?\n', '', final_func_code, flags=re.MULTILINE)
                final_func_code = re.sub(r'^import tools\..*?\n', '', final_func_code, flags=re.MULTILINE)

                # 4. 构建独立脚本内容
                script_content = f"""# -*- coding: utf-8 -*-
# Auto-generated Image Enhancement Script
# Generated by ChatImageEnhance
# Usage: python image_enhancement_script.py <input_dir> <output_dir>

import cv2
import numpy as np
import os
import argparse
import sys

# 设置标准输出编码为 UTF-8（避免 Windows 命令行乱码）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

{wrapper_class_code}
{final_func_code}

# Optimized Parameters
best_params = {repr(best_params)}

def batch_process(input_dir, output_dir):
    \"\"\"批量处理图像\"\"\"
    # ===================== 修复：自动创建输入/输出文件夹 =====================
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
        image = cv2.imread(img_path)
        if image is not None:
            try:
                enhanced = process(image, best_params)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, enhanced)
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
                
                st.download_button(
                    label="📦 导出为处理脚本",
                    data=script_content,
                    file_name="image_enhancement_script.py",
                    mime="text/x-python",
                    key=f"export_script_{id(msg)}_{index}"
                )
            else:
                st.button("📦 导出为处理脚本", disabled=True, help="需要生成处理代码和最优参数才能导出")

        prev_image = get_previous_img(index, ignore_test_mode=False)

        with st.container(border=False):
            comp_target = "原图"
            if prev_image is not None:
                comp_target = st.pills("对比对象", ["原图", "上一轮"], default="原图", required=True)

            image_comparison(
                get_thumbnail_img_wrapper(st.session_state['img_bgr'], 'b64') if comp_target == "原图" else get_thumbnail_img_wrapper(prev_image, 'b64'),
                get_thumbnail_img_wrapper(st.session_state['best_bgr'], 'b64'),
                get_thumbnail_size(st.session_state['best_bgr'], st.session_state['preview_img_max_side'])[1],
                comp_target,
                "最新"
            )

def generate_user_prompt(
    user_feedback: str,
    include_process: bool = False, 
    include_evaluate: bool = False, 
    step_by_step: bool = False
):
    last_assistant_msg = next(
        (
            m for m in reversed(st.session_state.messages[:-1]) 
            if m['role'] == 'assistant' and "image" in m and "test_mode" not in m
        ), 
        None
    )
    
    current_iter_prompt = f""
    if last_assistant_msg and not step_by_step:
        current_iter_prompt += f"--- 上一轮执行状态/系统回复 ---\n{last_assistant_msg['content']}\n"

        l_params = last_assistant_msg.get("best_params", {})
        l_eval = last_assistant_msg.get("eval_code", "")
        l_proc = last_assistant_msg.get("process_code", "")
        
        if l_eval and l_proc:
            if include_evaluate: current_iter_prompt += f"\n--- 上一轮使用的评价函数代码 ---\n```python\n{l_eval}\n```\n"
            if include_process: current_iter_prompt += f"\n--- 上一轮使用的图像处理代码 ---\n```python\n{l_proc}\n```\n"

        if l_params:
            current_iter_prompt += f"\n--- 上一轮 Optuna 搜索到的最优参数 ---\n{l_params}\n"

        current_iter_prompt += f"\n--- 本轮用户最新反馈/要求 ---\n{user_feedback}"
        # current_iter_prompt += "\n请仅基于全局目标、上一轮的状态和本次人类的最新反馈，修改评价指标、代码或 Optuna 参数范围。"
    else:
        current_iter_prompt += f"--- 用户要求 ---\n{user_feedback}"
    return current_iter_prompt