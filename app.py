import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile

from openai import OpenAI
from typing import Generator
from queue import Queue

from core.orchestrator import Orchestrator
from agents.coder import CoderAgent
from agents.evaluator import EvaluatorAgent
from components.optuna_callbacks import StOptunaCallback, StOptunaCallbackImg

st.set_page_config(
    page_title="AutoImageEnhance",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'mode' not in st.session_state:
    st.session_state['mode'] = "清晰锐化"
if 'models' not in st.session_state:
    st.session_state['models'] = []
if 'api_url' not in st.session_state:
    st.session_state['api_url'] = ""
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""
if 'has_api_key' not in st.session_state:
    st.session_state['has_api_key'] = False

@st.cache_resource
def get_openai_client(base_url, api_key):
    try:
        return OpenAI(base_url=base_url, api_key=api_key, max_retries=3)
    except Exception as e:
        print(e)
        return None

def clear_models():
    st.session_state.models = None

def get_models():
    if not st.session_state.api_url or (st.session_state.has_api_key and not st.session_state.api_key):
        return
    try:
        models = get_openai_client(st.session_state.api_url, st.session_state.api_key).models.list()
        if models:
            st.session_state.models = [model.id for model in models]
        else:
            st.session_state.models = None
    except Exception as e:
        print(e)
        pass

st.title("✨ AutoImageEnhance")
st.caption("LLM Agent + Optuna 全自动图像增强系统")
st.divider()

with st.sidebar:
    st.header("模型设置")
    s1c1, s1c2 = st.columns([0.001, 0.999])
    with s1c1: pass
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
    with s2c1: pass
    with s2c2:
        n_trials = st.slider("优化轮数", 5, 60, 15)
        # mode = st.selectbox("增强目标", [
        #     "自动增强", "清晰锐化", "低光增强", "去雾", "色彩饱和", "自定义"
        # ])
        mode = st.selectbox("增强目标", [
            "清晰锐化"
        ], key="mode")
        prompt = st.text_area("自定义描述", disabled = False if st.session_state['mode'] == "自定义" else True)

final_prompt = prompt if mode == "自定义" else mode

upload = st.file_uploader("上传图像", ["png", "jpg", "jpeg"])

if upload:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.subheader("原图")
    st.image(img_bgr, width="stretch", channels="BGR")

if st.button("🚀 启动自动增强", type="primary", disabled=not selected_model):
    if not upload or not final_prompt:
        st.warning("请上传图片并输入描述")
        st.stop()
    if not api_url:
        st.warning("请输入API URL")
        st.stop()

    if not selected_model:
        st.warning("未选择使用的模型")
        st.stop()
    
    client = get_openai_client(st.session_state.api_url, st.session_state.api_key)
    orch = Orchestrator(
        CoderAgent(
            client, selected_model
        ),
        EvaluatorAgent(
            client, selected_model
        )
    )

    with (main_status := st.status("运行中...", expanded=True)):
        # with (init_status := st.status("💿 流程初始化", state="running")):
        #     st.text("对输入的原始图像进行必要的计算")
        with (eva_status := st.status("📝 LLM 生成评价策略", state="error")):
            eva_message = st.chat_message('assistant')
        with (llm_status := st.status("🧠 LLM 生成增强策略", state="error")):
            chat_message = st.chat_message('assistant')
        with (optuna_status := st.status("🔬 Optuna 贝叶斯优化", state="error")):
            status_text = st.empty()
            progress_bar = st.progress(0)
            preview_tab, data_tab = st.tabs(["图像预览", "调优记录（最近10条）"])
            with preview_tab: best_img = st.empty()
            with data_tab: table_placeholder = st.empty()

        best_queue = Queue()
        callback = StOptunaCallbackImg(n_trials, progress_bar, status_text, table_placeholder, best_img, best_queue)

        best_bgr, best_params, log = None, None, None
        # best_bgr, best_params, log = orch.process(
        #     image=img_bgr,
        #     user_prompt=final_prompt,
        #     n_trials=n_trials,
        #     callbacks=[callback]
        # )
        evaluate_code_str = ''
        def eva_stream_wrapper():
            global eva_status, evaluate_code_str
            for t, body in orch.prepare_stream(
                image=img_bgr,
                user_prompt=final_prompt,
            ):
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

        def stream_wrapper():
            global best_bgr, best_params, log, llm_status, optuna_status #, init_status
            for t, body in orch.process_stream(
                image=img_bgr,
                evaluate_code_str=evaluate_code_str,
                best_queue=best_queue,
                user_prompt=final_prompt,
                n_trials=n_trials,
                callbacks=[callback]
            ):
                # if t == "INIT.FINISH":
                #     init_status.update(state="complete")
                if t == "CODE.START":
                    llm_status.update(state="running")
                elif t == "CODE.REASONING":
                    yield body
                elif t == "CODE.STREAM":
                    yield body
                elif t == "CODE.END":
                    llm_status.update(state="complete")
                    optuna_status.update(state="running")
                elif t == "FINISH":
                    optuna_status.update(state="complete")
                    best_bgr, best_params, log = body

        chat_message.write_stream(stream_wrapper())

    if best_bgr is None:
        st.error("运行失败！")
        st.stop()

    st.subheader("结果对比")
    c1, c2 = st.columns(2)
    with c1: st.image(img_bgr, caption="原图", channels="BGR")
    with c2: st.image(best_bgr, caption="增强后", channels="BGR")

    with st.expander("最优参数"):
        st.json(best_params)
    # st.dataframe(log)

    main_status.update(label="运行结束", state="complete")

    # 下载
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        best_bgr: np.ndarray
        succ, enc_img = cv2.imencode('.png', best_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        enc_img.tofile(f.name)
        with open(f.name, "rb") as tf:
            st.download_button("📥 保存结果", tf, "enhanced.png", "image/png")