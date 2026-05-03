import streamlit as st
import cv2
import numpy as np
import httpx
import yaml

from openai import OpenAI, DefaultHttpxClient
from queue import Queue
from streamlit_local_storage import LocalStorage

from core.orchestrator import Orchestrator
from core.evaluator import Evaluator
from core.searcher import Searcher

from agents.coder import CoderAgent
from agents.evaluator import EvaluatorAgent
from agents.planner import PlannerAgent
from agents.toolmaker import ToolMakerAgent

from components.optuna_callbacks import StOptunaCallbackImg
from components.tool_search import StSearch
from components.llm_response_handler import StStreamResHandler

from utils import get_executable_dir

localS = LocalStorage()

st.set_page_config(
    page_title="ChatImageEnhance",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'models' not in st.session_state:
    st.session_state['models'] = []
if 'api_url' not in st.session_state:
    st.session_state['api_url'] = localS.getItem('api_url') or ""
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = localS.getItem('api_key') or ""
if 'has_api_key' not in st.session_state:
    st.session_state['has_api_key'] = st.session_state.api_key != ""
if 'proxy_url' not in st.session_state:
    st.session_state['proxy_url'] = localS.getItem('proxy_url') or ""
if 'github_token' not in st.session_state:
    st.session_state['github_token'] = localS.getItem('github_token') or ""

@st.cache_resource
def get_openai_client(base_url: str, api_key: str, proxy_url: str):
    try:
        if proxy_url:
            try:
                client = DefaultHttpxClient(
                    transport=httpx.HTTPTransport(
                        proxy=proxy_url
                    )
                )
                return OpenAI(base_url=base_url, api_key=api_key, max_retries=0, http_client=client)
            except Exception as e:
               print(e)
        return OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    except Exception as e:
        print(e)
        return None

def clear_models():
    st.session_state.models = None

def get_models():
    if not st.session_state.api_url or (st.session_state.has_api_key and not st.session_state.api_key):
        return
    try:
        models = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url).models.list()
        if models:
            st.session_state.models = [model.id for model in models]
        else:
            st.session_state.models = None
    except Exception as e:
        print(e)
        pass

st.title("✨ ChatImageEnhance")
st.caption("LLM Agent + Optuna 人类在环图像增强系统")
st.divider()

with st.sidebar:
    st.header("设置")
    with st.expander("模型", expanded=True):
        api_url = st.text_input(
            "API URL (OpenAI 兼容端点)", 
            placeholder='https://api.openai.com/v1', 
            key="api_url",
            on_change=clear_models
        )
        api_key = ""
        if (has_api_key := st.toggle("API KEY?", key="has_api_key")):
            api_key = st.text_input("API KEY", type='password', key="api_key")
        proxy_url = st.text_input("HTTP 代理服务器", placeholder="http://localhost:7890", key="proxy_url")
        fetch_button = st.button("获取模型列表", disabled = True if not api_url else False, width="stretch")
        if fetch_button:
            get_models()
        
        # 添加无模型选项用于测试
        model_options = ["🚫 无模型 (测试模式)"] + (st.session_state.models or [])
        selected_model = st.selectbox(
            "模型", 
            options = model_options,
            index = 0
        )
        
        # 如果选择了无模型选项，将selected_model设为None
        if selected_model == "🚫 无模型 (测试模式)":
            selected_model = None

    with st.expander("代码检索", expanded=True):
        st.text("这是什么", help="缺少工具时，使用GitHub REST API检索相关的代码，需要填写Token才能使用")
        github_token = st.text_input("GitHub Token", type='password', key="github_token")
        search_steps_limit = st.slider(
            "搜索步骤数限制", 10, 100, 30, step=1, disabled=not github_token,
            help="限制最多 LLM 调用次数，防止难以找到时无限运行"
        )
        search_interval = st.slider(
            "强制步骤间隔 (秒)", 0.0, 60.0, 5.0, step=0.5, disabled=not github_token,
            help="强行在两次请求之间插入间隔，防止请求过于频繁"
        )
    
    with st.expander("处理", expanded=True):
        step_by_step = st.toggle(
            "处理时基于上一轮图像",
            help="不勾选时每次处理均在原图上进行，勾选后则对上一轮结果进行"
        )
        n_trials = st.slider("优化轮数", 5, 150, 15)

    if (cache_api := fetch_button and st.session_state.models) or github_token:
        with st.container(height=1, border=False):
            if cache_api:
                if api_url: localS.setItem("api_url", api_url, "locals_api_url")
                elif localS.getItem("api_url"): localS.deleteItem("api_url", "del_locals_api_url")

                if api_key: localS.setItem("api_key", api_key, "locals_api_key")
                elif localS.getItem("api_key"): localS.deleteItem("api_key", "del_locals_api_key")

                if proxy_url: localS.setItem("proxy_url", proxy_url, "locals_proxy_url")
                elif localS.getItem("proxy_url"): localS.deleteItem("proxy_url", "del_locals_proxy_url")

            if github_token: localS.setItem("github_token", github_token, "locals_github_token")
            elif localS.getItem("github_token"): localS.deleteItem("github_token", "del_locals_github_token")

upload = st.file_uploader("上传图像", ["png", "jpg", "jpeg"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'best_bgr' not in st.session_state:
    st.session_state['best_bgr'] = None

# 如果有历史结果，并在界面顶部展示原图与当前最佳进度的对比
if upload:
    evaluator = Evaluator(img_bgr)
    top_preview_placeholder = st.empty()
    
    def update_top_preview():
        """用于局部刷新顶部图像预览的函数"""
        with top_preview_placeholder.container():
            if st.session_state['best_bgr'] is not None:
                st.subheader("当前优化进度")
                c1, c2 = st.columns(2)
                with c1: st.image(img_bgr, caption="原图", channels="BGR")
                with c2: st.image(st.session_state['best_bgr'], caption="当前最新增强结果", channels="BGR")
                st.divider()
            else:
                st.subheader("原图")
                st.image(img_bgr, width="stretch", channels="BGR")
    
    update_top_preview()
else:
    st.session_state.messages.clear()
    st.session_state['best_bgr'] = None

@st.cache_data
def get_encoded_img(raw_array: np.ndarray):
    succ, enc_img = cv2.imencode('.png', raw_array, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return succ, enc_img.tobytes()

def render_message_content(msg, index):
    """提取内部渲染逻辑，供历史记录与最新消息复用"""
    st.markdown(msg["content"])
    if "image" in msg:
        prev_image = None
        if index > 0:
            # 向前查找最近一个包含图像的assistant消息
            for i in range(index - 1, -1, -1):
                prev_msg = st.session_state.messages[i]
                if prev_msg["role"] == "assistant" and "image" in prev_msg:
                    prev_image = prev_msg["image"]
                    break

        with st.container():
            comp_img_type = "原图"
            if prev_image:
                comp_img_type = st.radio("对比的图像", options=["原图", "上一轮"], horizontal=True)
            img_preview_c1, img_preview_c2 = st.columns(2)
            with img_preview_c1: 
                if comp_img_type == "原图":
                    st.image(img_bgr, caption="原图", channels="BGR")
                else:
                    st.image(prev_image, caption="上一轮优化结果", channels="BGR")
            with img_preview_c2: 
                st.image(msg["image"], caption="此轮优化结果", channels="BGR")

        with st.expander("🛠️ 查看此轮生成的代码与最优参数"):
            with st.expander("评价逻辑 (Evaluation Code)"):
                st.code(msg.get("eval_code", "# 无评价代码"), language="python")
            
            with st.expander("处理逻辑 (Process Code)"):
                st.code(msg.get("process_code", "# 无处理代码"), language="python")
            
            with st.expander("Optuna 最优参数组合"):
                st.json(msg.get("best_params", {}))
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
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
            
        with btn_col2:
            if msg["new_tool"]:
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

# --- 渲染历史聊天记录 ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        render_message_content(msg, i)

user_feedback = st.chat_input(
    '描述你的增强要求或对上轮结果的反馈\n（例如："这张图有些模糊，给我锐化一下" 或 "这版锐化过度了，稍微柔和一点"）', 
    disabled=not upload
)

if upload:
    if not st.session_state['messages']:
        start_analyze = False
        if st.button("💡 不知如何描述？让 AI 分析", key="ai_planner_btn", use_container_width=True, disabled=start_analyze):
            start_analyze = True
            # 1. 模拟用户发起了分析请求
            st.session_state.messages.append({"role": "user", "content": "请帮我分析这张图像的问题，并给出增强建议。"})
            with st.chat_message("user"):
                st.markdown("请帮我分析这张图像的问题，并给出增强建议。")

            # 2. 立即触发 AI 分析流
            with st.chat_message("assistant"):
                with st.status("🔎 AI 正在分析图像客观指标与视觉问题...", expanded=True) as plan_status:
                    client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
                    planner = PlannerAgent(client, selected_model)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    analyze_result = {}
                    plan_placeholder = st.empty()
                    
                    def plan_stream_wrapper():
                        global analyze_result
                        # 将评价指标一并传给 Planner
                        for t, body in planner.execute_stream(evaluator.get_profile_yaml(), img_rgb):
                            if t in ["STREAM.REASONING", "STREAM.CONTENT"]:
                                yield body
                            elif t == "FINISH":
                                analyze_result = body
                                
                    plan_placeholder.write_stream(plan_stream_wrapper())
                    plan_status.update(label="图像分析完成", state="complete", expanded=False)
                
                # 3. 解析 Planner 的输出并格式化为友好的对话消息
                if analyze_result:
                    summary = analyze_result.get('diagnosis_summary', '未得出明确总结')
                    suggestion = analyze_result.get('enhancement_prompt', '')
                    
                    response_text = f"**📊 图像诊断总结：**\n{summary}\n\n"
                    
                    if 'identified_issues' in analyze_result:
                        response_text += "**🔍 发现的具体问题：**\n"
                        response_text += "| 问题类型 | 严重度 | 依据 |\n"
                        response_text += "| --- | --- | --- |\n"
                        for issue in analyze_result['identified_issues']:
                            response_text += f"| {issue.get('issue_type', '未知')} | {issue.get('severity', '未知')} | {issue.get('evidence')}\n"
                    
                    response_text += f"\n**✨ 推荐增强提示词：**\n```text\n{suggestion}\n```\n"
                    response_text += "\n*💡 您可以直接复制上面的提示词发送给我，或在此基础上做出一定的调整*"
                    
                    # 渲染到界面并存入历史记录
                    new_msg = {"role": "assistant", "content": response_text}
                    st.session_state.messages.append(new_msg)
                    render_message_content(new_msg, len(st.session_state.messages) - 1)

if user_feedback:
    if not upload:
        st.warning("请先上传图片")
        st.stop()
    
    # 如果选择了模型但没有配置API URL，则警告
    if selected_model and not api_url:
        st.warning("请输入API URL")
        st.stop()
    
    # 如果选择了模型但未实际选择（即不是测试模式），则警告
    if selected_model is None and "messages" in st.session_state and len(st.session_state.messages) > 0:
        # 已经有历史记录说明之前用过模型，现在切换到无模型模式需要提示
        pass  # 允许在测试模式下继续
    
    # 1. 记录人类用户的输入
    user_feedback_msg = {"role": "user", "content": user_feedback}
    st.session_state.messages.append(user_feedback_msg)
    with st.chat_message(user_feedback_msg["role"]):
        render_message_content(user_feedback_msg, len(st.session_state.messages) - 1)

    def generate_user_prompt(include_process: bool = False, include_evaluate: bool = False, step_by_step: bool = False):
        last_assistant_msg = next(
            (
                m for m in reversed(st.session_state.messages[:-1]) 
                if m['role'] == 'assistant' and "image" in m
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

    # 2. 启动智能体响应
    with st.chat_message("assistant"):
        # 整合上下文策略：将初始目标、历史反馈和当前诉求组装给 LLM

        # 检查是否为无模型测试模式
        if not selected_model:
            st.info("🧪 当前处于【无模型测试模式】，直接返回原图。")
            
            # 直接使用原图作为"增强结果"
            best_bgr = img_bgr.copy()
            best_params = {"mode": "test_no_llm", "info": "无模型测试模式，未进行任何处理"}
            
            # 更新会话状态
            st.session_state['best_bgr'] = best_bgr
            
            # 将系统的回应和新图像记入历史记录
            st.session_state.messages.append({
                "role": "assistant",
                "content": "✅ 【测试模式】已返回原图。请选择一个有效的模型以启用真正的 AI 图像增强功能。",
                "image": best_bgr,
                "eval_code": "# 无模型测试模式，未生成评价代码",
                "process_code": "# 无模型测试模式，未生成处理代码",
                "best_params": best_params
            })
            
            st.rerun()

        # 检查是否为无模型测试模式
        if not selected_model:
            st.info("🧪 当前处于【无模型测试模式】，直接返回原图。")
            
            # 直接使用原图作为"增强结果"
            best_bgr = img_bgr.copy()
            best_params = {"mode": "test_no_llm", "info": "无模型测试模式，未进行任何处理"}
            
            # 更新会话状态
            st.session_state['best_bgr'] = best_bgr
            
            # 将系统的回应和新图像记入历史记录
            st.session_state.messages.append({
                "role": "assistant",
                "content": "✅ 【测试模式】已返回原图。请选择一个有效的模型以启用真正的 AI 图像增强功能。",
                "image": best_bgr,
                "eval_code": "# 无模型测试模式，未生成评价代码",
                "process_code": "# 无模型测试模式，未生成处理代码",
                "best_params": best_params
            })
            
            st.rerun()

        client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
        orch = Orchestrator(
            CoderAgent(client, selected_model),
            EvaluatorAgent(client, selected_model),
            ToolMakerAgent(client, selected_model)
        )

        with (main_status := st.status("🛠️ 根据反馈调整并运行...", expanded=True)):
            with (eva_status := st.status("📝 LLM 调整评价策略", state="error")):
                eva_thinking_container = st.container(border=False)
            with (main_container := st.container(border=False)):
                with (code_status := st.status("🧠 LLM 调整增强代码", state="error")):
                    code_thinking_container = st.container(border=False)
            with (optuna_status := st.status("🔬 Optuna 重新调优", state="error")):
                status_text = st.empty()
                progress_bar = st.progress(0)
                preview_tab, data_tab = st.tabs(["实时图像预览", "调优记录"])
                with preview_tab: best_img = st.empty()
                with data_tab: table_placeholder = st.empty()

            best_queue = Queue()
            callback = StOptunaCallbackImg(n_trials, progress_bar, status_text, table_placeholder, best_img, best_queue)

            best_bgr, best_params, log = None, None, None
            evaluate_code_str = ''
            process_code_str = ''
            new_tool: dict | None = None

            evaluate_handler = StStreamResHandler(eva_status, eva_thinking_container)

            for t, body in orch.prepare_stream(image=img_bgr, user_prompt=generate_user_prompt(True, True, step_by_step)):
                if t == "CODE_EVALUATE.START":
                    eva_status.update(state="running")
                elif t == "CODE_EVALUATE.REASONING":
                    evaluate_handler.thinking_chunk(body)
                elif t == "CODE_EVALUATE.STREAM":
                    evaluate_handler.content_chunk(body)
                    evaluate_handler.thinking_end()
                elif t == "CODE_EVALUATE.END":
                    evaluate_handler.content_end()
                elif t == "FINISH":
                    eva_status.update(state="complete")
                    evaluate_code_str = body
                    evaluate_handler.set_content(f"```python\n{evaluate_code_str}\n```")

            coding_finish = False

            while not coding_finish:
                tool_request = ''
                tool_status = None
                search_result = dict()
                coder_handler = StStreamResHandler(code_status, code_thinking_container)

                orch.coder.rebuild_system_prompt()
                for t, body in orch.process_stream(
                    image=st.session_state['best_bgr'] if step_by_step and st.session_state['best_bgr'] else img_bgr,
                    evaluate_code_str=evaluate_code_str,
                    best_queue=best_queue,
                    user_prompt=generate_user_prompt(True, True, step_by_step),
                    n_trials=n_trials,
                    callbacks=[callback]
                ):
                    if t == "CODE.START":
                        pass
                    elif t == "CODE.REASONING":
                        coder_handler.thinking_chunk(body)
                    elif t == "CODE.STREAM":
                        coder_handler.content_chunk(body)
                        coder_handler.thinking_end()
                    elif t == "CODE.END":
                        coder_handler.content_end()
                        process_code_str = body
                    elif t == "OPTUNA.START":
                        optuna_status.update(state="running")
                    elif t == "FINISH":
                        optuna_status.update(state="complete")
                        best_bgr, best_params, log = body

                        coding_finish = True
                    
                    elif t == "TOOL_REQUEST":
                        code_status.update(state='error')
                        tool_request = body['description']

                        if github_token:
                            with main_container:
                                with (tool_status := st.status("⌨️ LLM 编写额外工具", state="error")):
                                    search_container = st.container(border=False)

                                tool_status.update(state='running')
                                if search_container:
                                    if github_token:
                                        searcher = Searcher(github_token, client, selected_model)
                                        search_result: dict = StSearch(
                                            searcher, tool_request, search_container, search_steps_limit, search_interval
                                        )
                        break
                
                if tool_request:
                    with main_container:
                        if not tool_status:
                            tool_status = st.status("⌨️ LLM 编写额外工具", state="error")

                        with tool_status:
                            with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
                                toolmaker_container = st.container(border=False)

                    toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
                        
                    for t, body in orch.toolmaker_stream(tool_request, search_result):
                        if t == "CODE_TOOL.STREAM":
                            toolmaker_handler.content_chunk(body)
                            toolmaker_handler.thinking_end()
                        elif t == "CODE_TOOL.REASONING":
                            toolmaker_handler.thinking_chunk(body)
                        elif t == "CODE_TOOL.END":
                            toolmaker_handler.content_end()
                        elif t == "ERROR_RETRY":
                            with main_container:
                                with tool_status:
                                    with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
                                        toolmaker_container = st.container(border=False)
                            
                            toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
                        elif t == "FINISH":
                            new_tool = body

        # --- 收尾与状态更新 ---
        if best_bgr is None:
            st.error("此轮运行失败，请尝试重新输入或更改要求。")
            st.stop()
        else:
            main_status.update(label="本轮调整结束", state="complete")
            st.session_state['best_bgr'] = best_bgr

            new_msg = {
                "role": "assistant",
                "content": "已完成本轮调优。请查看图像，如果需要进一步调整（如：增加亮度、减少对比度），请直接告诉我。",
                "image": best_bgr,
                "eval_code": evaluate_code_str,
                "process_code": process_code_str,
                "best_params": best_params,
            }
            if new_tool: new_msg["new_tool"] = new_tool
            st.session_state.messages.append(new_msg)

            render_message_content(new_msg, len(st.session_state.messages) - 1)
            update_top_preview()