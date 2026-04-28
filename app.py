import streamlit as st
import cv2
import numpy as np
import httpx

from openai import OpenAI, DefaultHttpxClient
from typing import Generator
from queue import Queue
from pathlib import Path

from core.orchestrator import Orchestrator
from core.evaluator import Evaluator
from agents.coder import CoderAgent
from agents.evaluator import EvaluatorAgent
from agents.planner import PlannerAgent
from components.optuna_callbacks import StOptunaCallback, StOptunaCallbackImg

st.set_page_config(
    page_title="ChatImageEnhance",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'models' not in st.session_state:
    st.session_state['models'] = []
if 'api_url' not in st.session_state:
    st.session_state['api_url'] = ""
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""
if 'has_api_key' not in st.session_state:
    st.session_state['has_api_key'] = False

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
    st.header("模型设置")
    s1c1, s1c2 = st.columns([0.001, 0.999])
    with s1c2:
        api_url = st.text_input(
            "API URL (OpenAI 兼容)", 
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
        selected_model = st.selectbox(
            "模型", 
            options = st.session_state.models,
            index = 0 if st.session_state.models else None
        )
    
    st.header("处理设置")
    s2c1, s2c2 = st.columns([0.001, 0.999])
    with s2c2:
        n_trials = st.slider("优化轮数", 5, 150, 15)

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
    if st.session_state['best_bgr'] is not None:
        st.subheader("当前优化进度")
        c1, c2 = st.columns(2)
        with c1: st.image(img_bgr, caption="原图", channels="BGR")
        with c2: st.image(st.session_state['best_bgr'], caption="当前最新增强结果", channels="BGR")
        st.divider()
    else:
        st.subheader("原图")
        st.image(img_bgr, width="stretch", channels="BGR")


# --- 渲染历史聊天记录 ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"], channels="BGR", caption="此轮优化结果")

            with st.expander("🛠️ 查看此轮生成的代码与最优参数"):
                with st.expander("评价逻辑 (Evaluation Code)"):
                    st.code(msg.get("eval_code", "# 无评价代码"), language="python")
                
                with st.expander("处理逻辑 (Process Code)"):
                    st.code(msg.get("process_code", "# 无处理代码"), language="python")
                
                with st.expander("Optuna 最优参数组合"):
                    st.json(msg.get("best_params", {}))
            
            # 从内存直接提供历史版本的下载
            succ, enc_img = cv2.imencode('.png', msg["image"], [cv2.IMWRITE_PNG_COMPRESSION, 2])
            if succ:
                st.download_button(
                    label="📥 保存此版本", 
                    data=enc_img.tobytes(), 
                    file_name=f"enhanced_history_{i}.png", 
                    mime="image/png", 
                    key=f"dl_history_{i}"
                )

user_feedback = st.chat_input(
    '描述你的增强要求或对上轮结果的反馈\n（例如："这张图有些模糊，给我锐化一下" 或 "这版锐化过度了，稍微柔和一点"）', 
    disabled=not selected_model
)

if upload and selected_model:
    if not st.session_state['messages']:
        if st.button("💡 不知如何描述？让 AI 分析", key="ai_planner_btn", use_container_width=True):
            # 1. 模拟用户发起了分析请求
            st.session_state.messages.append({"role": "user", "content": "请帮我分析这张图像的问题，并给出增强建议。"})
            
            # 2. 立即触发 AI 分析流
            with st.chat_message("assistant"):
                with st.status("🔎 AI 正在分析图像客观指标与视觉问题...", expanded=True) as plan_status:
                    client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
                    planner = PlannerAgent(client, selected_model)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    evaluator = Evaluator(img_bgr)
                    
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
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()

if user_feedback:
    if not upload:
        st.warning("请先上传图片")
        st.stop()
    if not api_url:
        st.warning("请输入API URL")
        st.stop()
    if not selected_model:
        st.warning("未选择使用的模型")
        st.stop()

    # 1. 记录人类用户的输入
    st.session_state.messages.append({"role": "user", "content": user_feedback})
    with st.chat_message("user"):
        st.markdown(user_feedback)

    # 2. 启动智能体响应
    with st.chat_message("assistant"):
        # 整合上下文策略：将初始目标、历史反馈和当前诉求组装给 LLM
        last_assistant_msg = next(
            (
                m for m in reversed(st.session_state.messages[:-1]) 
                if m['role'] == 'assistant' and "image" in m
            ), 
            None
        )
        
        current_iter_prompt = f""
        if last_assistant_msg:
            current_iter_prompt += f"--- 上一轮执行状态/系统回复 ---\n{last_assistant_msg['content']}\n"

            l_params = last_assistant_msg.get("best_params", {})
            l_eval = last_assistant_msg.get("eval_code", "")
            l_proc = last_assistant_msg.get("process_code", "")
            
            if l_eval and l_proc:
                current_iter_prompt += f"\n--- 上一轮使用的评价函数代码 ---\n```python\n{l_eval}\n```\n"
                current_iter_prompt += f"\n--- 上一轮使用的图像处理代码 ---\n```python\n{l_proc}\n```\n"

            if l_params:
                current_iter_prompt += f"\n--- 上一轮 Optuna 搜索到的最优参数 ---\n{l_params}\n"

            current_iter_prompt += f"\n--- 本轮用户最新反馈/要求 ---\n{user_feedback}"
            # current_iter_prompt += "\n请仅基于全局目标、上一轮的状态和本次人类的最新反馈，修改评价指标、代码或 Optuna 参数范围。"
        else:
            current_iter_prompt += f"--- 用户要求 ---\n{user_feedback}"
        # ==========================================

        client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
        orch = Orchestrator(
            CoderAgent(client, selected_model),
            EvaluatorAgent(client, selected_model)
        )

        with (main_status := st.status("🛠️ 根据反馈调整并运行...", expanded=True)):
            with (eva_status := st.status("📝 LLM 调整评价策略", state="error")):
                eva_message = st.empty()
            with (llm_status := st.status("🧠 LLM 调整增强代码", state="error")):
                chat_message = st.empty()
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

            # 执行 Evaluator 流
            def eva_stream_wrapper():
                global eva_status, evaluate_code_str
                # 注意这里传入 current_iter_prompt
                for t, body in orch.prepare_stream(image=img_bgr, user_prompt=current_iter_prompt):
                    if t == "CODE_EVALUATE.START":
                        eva_status.update(state="running")
                    elif t == "CODE_EVALUATE.REASONING":
                        yield body
                    elif t == "CODE_EVALUATE.STREAM":
                        yield body
                    elif t == "CODE_EVALUATE.END":
                        pass
                    elif t == "FINISH":
                        eva_status.update(state="complete")
                        evaluate_code_str = body
            
            eva_message.write_stream(eva_stream_wrapper())

            process_code_str = ""

            # 执行 Coder 与 Optuna 流
            def stream_wrapper():
                global best_bgr, best_params, log, llm_status, optuna_status, process_code_str
                # 同样传入包含人类反馈的 current_iter_prompt
                for t, body in orch.process_stream(
                    image=img_bgr,
                    evaluate_code_str=evaluate_code_str,
                    best_queue=best_queue,
                    user_prompt=current_iter_prompt,
                    n_trials=n_trials,
                    callbacks=[callback]
                ):
                    if t == "CODE.START":
                        llm_status.update(state="running")
                    elif t == "CODE.REASONING":
                        yield body
                    elif t == "CODE.STREAM":
                        yield body
                    elif t == "CODE.END":
                        llm_status.update(state="complete")
                        optuna_status.update(state="running")
                        process_code_str = body
                    elif t == "FINISH":
                        optuna_status.update(state="complete")
                        best_bgr, best_params, log = body

            chat_message.write_stream(stream_wrapper())

        # --- 收尾与状态更新 ---
        if best_bgr is None:
            st.error("此轮运行失败，请尝试重新输入或更改要求。")
            st.stop()
        else:
            main_status.update(label="本轮调整结束", state="complete")
            st.session_state['best_bgr'] = best_bgr

            with st.expander("最优参数 (用于调试)"):
                st.json(best_params)

            # 将系统的回应和新图像记入历史记录
            st.session_state.messages.append({
                "role": "assistant",
                "content": "已完成本轮调优。请查看图像，如果需要进一步调整（如：增加亮度、减少对比度），请直接告诉我。",
                "image": best_bgr,
                "eval_code": evaluate_code_str,
                "process_code": process_code_str,
                "best_params": best_params
            })

            st.rerun()