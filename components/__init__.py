import streamlit as st
import yaml

from components.image_comparison import image_comparison
from utils import *

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
        prev_image = get_previous_img(index, ignore_test_mode=False)

        with st.container(border=True):
            comp_target = "原图"
            if prev_image is not None:
                comp_target = st.radio("对比对象", ["原图", "上一轮"], horizontal=True)

            image_comparison(
                get_thumbnail_img_wrapper(st.session_state['img_bgr'], 'b64') if comp_target == "原图" else get_thumbnail_img_wrapper(prev_image, 'b64'),
                get_thumbnail_img_wrapper(st.session_state['best_bgr'], 'b64'),
                get_thumbnail_size(st.session_state['best_bgr'], st.session_state['preview_img_max_side'])[1],
                comp_target,
                "最新"
            )

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