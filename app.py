import streamlit as st
import cv2
import numpy as np
import os
import gc
import torch

from utils import *
from constants import *

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_HOME"] = str(get_executable_dir() / 'caches/model_assets/torch')
os.environ["HF_HOME"] = str(get_executable_dir() / 'caches/model_assets/huggingface')
os.environ["MODELSCOPE_CACHE"] = str(get_executable_dir() / "caches/model_assets/modelscope")
os.environ["PIP_CACHE_DIR"] = str(get_executable_dir() / 'caches/pip')

import transformers
transformers.logging.set_verbosity_error()

from collections import deque
from streamlit_local_storage import LocalStorage

from components.optuna_callbacks import StOptunaCallbackImg
from components.tool_search import StEnrichFindings, StSearch
from components.llm_response_handler import StStreamResHandler
from components.image_analyze import image_analyze
from components.tools_playground import render_playground
from components.debug_toolmaker import render_toolmaker
from components import get_thumbnail_img_wrapper, render_message_content, get_previous_img, generate_user_prompt, extract_funcs

from core.orchestrator import Orchestrator
from core.evaluator import Evaluator
from core.searcher import Searcher

from agents.coder import CoderAgent
from agents.evaluator import EvaluatorAgent
from agents.toolmaker import ToolMakerAgent

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
if 'huggingface_token' not in st.session_state:
    st.session_state['huggingface_token'] = localS.getItem('huggingface_token') or ""
if 'modelscope_token' not in st.session_state:
    st.session_state['modelscope_token'] = localS.getItem('modelscope_token') or ""
if 'reasoning_effort' not in st.session_state:
    st.session_state['reasoning_effort'] = None
if 'process_img_max_side' not in st.session_state:
    st.session_state['process_img_max_side'] = 1500
if 'device_learning_process' not in st.session_state:
    st.session_state['device_learning_process'] = "cpu"
if 'process_profile' not in st.session_state:
    st.session_state['process_profile'] = "balanced"
if 'process_operator_preference' not in st.session_state:
    st.session_state['process_operator_preference'] = "prefer_traditional"
if 'img_bgr' not in st.session_state:
    st.session_state['img_bgr'] = None
if 'running' not in st.session_state:
    st.session_state['running'] = False

st.title("✨ ChatImageEnhance")
st.caption("LLM Agent + Optuna 人类在环图像增强系统")
st.divider()

@st.cache_resource
def get_cv2_inter_mapping() -> dict[int, str]:
    return {
        getattr(cv2, name): name
        for name in dir(cv2) 
        if name.startswith('INTER_') and not any(x in name for x in ['MAX', 'TAB', 'BITS'])
    }

@st.cache_resource
def get_model_cache():
    return dict()

PROCESS_OPERATOR_PREFERENCES = [
    "traditional_only",
    "prefer_traditional",
    "prefer_learning",
    "learning_only"
]

PROCESS_OPERATOR_PREFERENCE_LABELS = {
    "traditional_only": "仅传统",
    "prefer_traditional": "偏好传统",
    "prefer_learning": "偏好深度学习",
    "learning_only": "仅深度学习",
}

def normalize_process_operator_preference(
    value: str | None,
    allow_learning_process: bool
) -> str:
    mode = str(value or "").strip().lower()
    alias = {
        "traditional_only": "traditional_only",
        "only_traditional": "traditional_only",
        "traditional": "traditional_only",
        "仅传统": "traditional_only",
        "prefer_traditional": "prefer_traditional",
        "traditional_preferred": "prefer_traditional",
        "偏好传统": "prefer_traditional",
        "prefer_learning": "prefer_learning",
        "prefer_deep_learning": "prefer_learning",
        "偏好深度学习": "prefer_learning",
        "learning_only": "learning_only",
        "only_learning": "learning_only",
        "only_deep_learning": "learning_only",
        "deep_learning_only": "learning_only",
        "仅深度学习": "learning_only",
    }
    normalized = alias.get(mode, "prefer_traditional")
    if not allow_learning_process:
        return "traditional_only"
    return normalized

def get_process_operator_preference_label(value: str) -> str:
    return PROCESS_OPERATOR_PREFERENCE_LABELS.get(value, value)

def build_runtime_hint(
    allow_learning_process: bool,
    process_device: str,
    process_profile: str,
    device_info_text: str,
    process_operator_preference: str | None = None
) -> str:
    lines = [
        "--- 运行时约束 ---",
        f"深度学习处理: {'enabled' if allow_learning_process else 'disabled'}",
    ]
    if process_operator_preference:
        lines.append(
            f"处理算子偏好: {get_process_operator_preference_label(process_operator_preference)}"
        )
    lines.extend([
        f"处理设备偏好: {process_device}",
        f"性能档位偏好: {process_profile}",
        "设备信息:",
        device_info_text,
        "---",
    ])
    return "\n".join(lines).strip("\n")

def get_allowed_search_sources(
    allow_learning_process: bool,
    operator_preference: str
) -> tuple[str, ...]:
    if (not allow_learning_process) or operator_preference == "traditional_only":
        return ("github",)
    return ("github", "huggingface", "modelscope")

def _detect_device_type(obj) -> str | None:
    """尝试从各种缓存对象中推断设备类型，容错处理。"""
    try:
        # nn.Module: 直接取 parameters
        if hasattr(obj, 'parameters'):
            param = next(obj.parameters(), None)
            if param is not None:
                return param.device.type
        # dict 形式: 可能含 'model' 或其他 nn.Module 值
        if isinstance(obj, dict):
            for v in obj.values():
                if hasattr(v, 'parameters'):
                    param = next(v.parameters(), None)
                    if param is not None:
                        return param.device.type
        # pipeline 对象: 可能有 device / device_name 属性
        device_attr = getattr(obj, 'device', None) or getattr(obj, 'device_name', None)
        if device_attr is not None:
            device_str = str(device_attr).strip().lower()
            if device_str and device_str != 'cpu':
                return device_str.split(':')[0]
    except Exception:
        pass
    return None

def unload_models():
    cache = get_model_cache()
    devices = set()
    for value in cache.values():
        device_type = _detect_device_type(value)
        if device_type and device_type != 'cpu':
            devices.add(device_type)
    cache.clear()
    gc.collect()
    for device in devices:
        if (device_module := getattr(torch, device, None)):
            if (empty_cache := getattr(device_module, 'empty_cache', None)):
                try:
                    empty_cache()
                except Exception:
                    pass
    # 兜底: 始终尝试清理 CUDA/MPS/XPU
    for backend in ('cuda', 'mps', 'xpu'):
        try:
            mod = getattr(torch, backend, None)
            if mod and hasattr(mod, 'empty_cache'):
                mod.empty_cache()
        except Exception:
            pass

with st.sidebar:
    st.header("设置")
    with st.expander("基本", expanded=True):
        st.pills(
            "执行模式", ["Chat", "Playground", "ToolMaker"], 
            required=True, default="Chat", key='ui_scene',
            help="选择Playground以查看并试用所有可用的算子"
        )
        enable_learning = st.toggle('启用深度学习')
        if enable_learning:
            with st.container(border=True):
                if st.toggle("用于评价", help="约占用3GB内存", key='enable_learning_evaluator'):
                    st.selectbox("评价模型运行设备", get_available_devices(), key='device_learning_evaluator')
                if st.toggle("用于处理", key='enable_learning_process'):
                    st.selectbox("处理模型运行设备", get_available_devices(), key='device_learning_process')
                    st.selectbox(
                        "处理显存策略",
                        ["fast", "balanced", "low_memory"],
                        key='process_profile',
                        format_func=lambda x: {
                            "fast": "高速",
                            "balanced": "平衡（默认）",
                            "low_memory": "低占用"
                        }.get(x, x),
                        help="用于高显存参数（如 tile_size）的默认倾向。实际回落由运行时自动完成。"
                    )
                    current_process_operator_preference = normalize_process_operator_preference(
                        st.session_state.get("process_operator_preference"),
                        True
                    )
                    st.selectbox(
                        "处理算子偏好",
                        PROCESS_OPERATOR_PREFERENCES,
                        index=PROCESS_OPERATOR_PREFERENCES.index(current_process_operator_preference),
                        key='process_operator_preference',
                        format_func=get_process_operator_preference_label,
                        help="仅档位会在Schema注入前过滤算子；偏好档位会保留全量算子并在提示词中表达偏好。"
                    )
        else:
            st.session_state['enable_learning_evaluator'] = False
            st.session_state['enable_learning_process'] = False
            st.session_state['device_learning_process'] = "cpu"
            st.session_state['process_profile'] = "balanced"
            if st.button("释放预载入模型", help="释放预载入到内存的模型", width='stretch', disabled=len(get_model_cache()) == 0):
                unload_models()

    with st.expander("模型", expanded=True):
        api_url = st.text_input(
            "API URL (OpenAI 兼容端点)", 
            placeholder='https://api.openai.com/v1', 
            key="api_url",
            on_change=clear_models
        )
        api_key = ""
        if (has_api_key := st.toggle("API KEY?", key="has_api_key")):
            api_key = st.text_input("API KEY", type='password', key="api_key", on_change=get_openai_client.clear)
        proxy_url = st.text_input("HTTP 代理服务器", placeholder="http://localhost:7890", key="proxy_url", on_change=get_openai_client.clear)
        fetch_button = st.button("获取模型列表", disabled = True if not api_url else False, width="stretch")
        if fetch_button:
            get_models()
        
        # 添加无模型选项用于测试
        model_options = [DEBUG_MODEL_NAME] + (st.session_state.models or [])
        selected_model = st.selectbox(
            "模型", 
            options = model_options,
            index = 0,
            key="selected_model",
            disabled=st.session_state.models is None or len(st.session_state.models) == 0
        )

        is_visual_model = st.toggle("该模型支持视觉输入", key="is_visual_model")

        st.slider("代码问题最多重试次数", value=3, min_value=3, max_value=10, key="llm_coding_max_retries")

        with st.expander("高级"):
            reasoning_effort = st.selectbox(
                "推理努力 Reasoning Effort", ['default', 'minimal', 'low', 'medium', 'high', 'xhigh'],
                help="请查询你使用的模型是否支持该字段。部分模型不支持minimal和xhigh。"
            )
            if reasoning_effort == "default":
                reasoning_effort = None
            st.session_state['reasoning_effort'] = reasoning_effort

    with st.expander("预览", expanded=True):
        preview_img_max_side = st.slider(
            "预览图像最长边 (px)", 
            300, 4000, 800, step=25, 
            on_change=get_thumbnail_img.clear,
            help="通过缩小预览图像尺寸提高加载速度并降低内存使用",
            key='preview_img_max_side'
        )

        inter_mapping = get_cv2_inter_mapping()
        inter_options = list(inter_mapping.keys())

        preview_img_scale = st.selectbox(
            "预览图像缩小算法", inter_options, 
            format_func=inter_mapping.get, 
            on_change=get_thumbnail_img.clear,
            key='preview_img_scale'
        )

    with st.expander("代码检索", expanded=True):
        st.text("这是什么", help="缺少工具时，可通过 GitHub / HuggingFace / ModelScope 检索相关代码或模型。")
        github_token = st.text_input("GitHub Token", type='password', key="github_token")
        huggingface_token = st.text_input("HuggingFace Token", type='password', key="huggingface_token")
        modelscope_token = st.text_input("ModelScope Token", type='password', key="modelscope_token")
        search_steps_limit = st.slider(
            "搜索步骤数限制", 10, 100, 30, step=1,
            help="限制最多 LLM 调用次数，防止难以找到时无限运行",
            key="search_steps_limit"
        )
        search_interval = st.slider(
            "强制步骤间隔 (秒)", 0.0, 60.0, 5.0, step=0.5,
            help="强行在两次请求之间插入间隔，防止请求过于频繁",
            key="search_interval"
        )
    
    with st.expander("处理", expanded=True):
        n_trials = st.slider("优化轮数", 5, 150, 15)
        
        # ===== [新增] 自适应优化设置面板 =====
        with st.expander("自适应优化设置", expanded=False):
            enable_early_stop = st.toggle(
                "启用自适应早停",
                value=True,
                help="当优化结果收敛时自动提前终止，节省时间"
            )
            patience = st.slider(
                "收敛耐心值",
                5, 30, 10,
                help="连续多少个trial没有显著改进时触发早停",
                disabled=not enable_early_stop
            )
            min_trials = st.slider(
                "最少优化轮数",
                3, 20, 5,
                help="至少执行多少轮后才允许早停",
                disabled=not enable_early_stop
            )
            improvement_threshold = st.slider(
                "改进阈值",
                0.001, 0.1, 0.01, 0.001,
                help="相对改进小于此值视为无显著改进",
                disabled=not enable_early_stop,
                format="%.3f"
            )
        
        step_by_step = st.toggle(
            "基于上一轮图像而非原图",
            help="不勾选时每次处理均在原图上进行，勾选后则对上一轮结果进行"
        )
        low_res_process = st.toggle(
            "使用低分辨率图像调优",
            help="使用低分辨率图像调优以提高速度，但可能影响最终效果",
            key="low_res_process"
        )
        if low_res_process:
            with st.container(border=True):
                process_img_max_side = st.slider(
                    "图像最长边 (px)", 
                    600, 4000, 1500, step=25,
                    key="process_img_max_side"
                )

    if (cache_api := fetch_button and st.session_state.models) or github_token or huggingface_token or modelscope_token:
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

            if huggingface_token: localS.setItem("huggingface_token", huggingface_token, "locals_huggingface_token")
            elif localS.getItem("huggingface_token"): localS.deleteItem("huggingface_token", "del_locals_huggingface_token")

            if modelscope_token: localS.setItem("modelscope_token", modelscope_token, "locals_modelscope_token")
            elif localS.getItem("modelscope_token"): localS.deleteItem("modelscope_token", "del_locals_modelscope_token")

def get_orchestrator():
    allow_learning_process = bool(st.session_state.get("enable_learning_process", False))
    process_operator_preference = normalize_process_operator_preference(
        st.session_state.get("process_operator_preference"),
        allow_learning_process
    )
    process_device = (
        st.session_state.get('device_learning_process', 'cpu')
        if allow_learning_process else 'cpu'
    )
    process_profile = (
        st.session_state.get('process_profile', 'balanced')
        if allow_learning_process else 'balanced'
    )
    toolmaker_allow_learning = (
        allow_learning_process and process_operator_preference != "traditional_only"
    )
    device_info_text = format_device_info_for_prompt(get_device_info_subprocess())

    client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
    orch = Orchestrator(
        CoderAgent(
            client, selected_model, 
            reasoning_effort=st.session_state.reasoning_effort, 
            low_res=low_res_process,
            allow_learning=allow_learning_process,
            operator_preference=process_operator_preference
        ),
        EvaluatorAgent(
            client, selected_model, 
            reasoning_effort=st.session_state.reasoning_effort,
            allow_learning=st.session_state.enable_learning_evaluator
        ),
        ToolMakerAgent(
            client, selected_model,
            reasoning_effort=st.session_state.reasoning_effort,
            allow_learning=toolmaker_allow_learning
        ),
        allow_learning_process=allow_learning_process,
        process_device=process_device,
        process_profile=process_profile,
        device_info=device_info_text,
        max_llm_retries=st.session_state['llm_coding_max_retries']
    )
    return client, orch

if st.session_state.ui_scene in ("ToolMaker", "Chat"):
    allow_learning_process = bool(st.session_state.get("enable_learning_process", False))
    process_operator_preference = normalize_process_operator_preference(
        st.session_state.get("process_operator_preference"),
        allow_learning_process
    )
    process_device = (
        st.session_state.get('device_learning_process', 'cpu')
        if allow_learning_process else 'cpu'
    )
    process_profile = (
        st.session_state.get('process_profile', 'balanced')
        if allow_learning_process else 'balanced'
    )
    device_info_text = format_device_info_for_prompt(get_device_info_subprocess())
    runtime_hint = build_runtime_hint(
        allow_learning_process,
        process_device,
        process_profile,
        device_info_text,
        process_operator_preference
    )
    toolmaker_runtime_hint = build_runtime_hint(
        allow_learning_process,
        process_device,
        process_profile,
        device_info_text
    )

if st.session_state.ui_scene == "ToolMaker":
    render_toolmaker(get_orchestrator()[1])
    st.stop()

upload = st.file_uploader("上传图像", ["png", "jpg", "jpeg"])

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'best_bgr' not in st.session_state:
    st.session_state['best_bgr'] = None
if 'evaluator' not in st.session_state:
    st.session_state['evaluator'] = None
if 'ui_scene' not in st.session_state:
    st.session_state['ui_scene'] = ""

@st.cache_resource
def get_evaluator(raw_array: np.ndarray):
    return Evaluator(raw_array, get_model_cache())

@st.cache_resource
def get_thumb_evaluator(raw_array: np.ndarray, size: tuple[int, int]):
    return Evaluator(raw_array, get_model_cache())

def append_run_error_message(error_text: str):
    error_text = str(error_text or "运行失败，未返回详细错误。").strip()
    summary = error_text
    details = ""
    marker = "\n\n错误详情：\n"
    if marker in error_text:
        summary, details = error_text.split(marker, maxsplit=1)
        summary = summary.strip()
        details = details.strip()

    error_msg = {
        "role": "assistant",
        "content": summary,
        "error": True,
        "error_details": details,
    }
    st.session_state.messages.append(error_msg)
    render_message_content(error_msg, len(st.session_state.messages) - 1)

# 如果有历史结果，并在界面顶部展示原图与当前最佳进度的对比
if upload:
    img_bgr = load_bgr_img_from_file(upload)
    st.session_state['img_bgr'] = img_bgr
    img_bgr_preview_bytes = get_thumbnail_img_wrapper(img_bgr, 'binary')

    top_preview_placeholder = st.empty()

    st.subheader("原图")
    st.image(
        img_bgr_preview_bytes, width="stretch", 
        caption=f"{img_bgr.shape[1]}x{img_bgr.shape[0]}, {img_bgr.shape[2]} Channel(s)"
    )
    st.divider()

    if st.session_state.ui_scene == "Chat":
        wait_init = st.empty()
        with wait_init:
            evaluator = get_evaluator(img_bgr)
            st.session_state['evaluator'] = evaluator
            # st.session_state['process_evaluator'] = evaluator

            # if st.session_state.low_res_process:
            #     resized, thumb_size = get_thumbnail_size(img_bgr, st.session_state.process_img_max_side)
            #     if resized:
            #         st.session_state['process_evaluator'] = get_thumb_evaluator(
            #             img_bgr, thumb_size
            #         )

        if st.session_state['enable_learning_evaluator']:
            with wait_init:
                with st.spinner("正在载入模型，耗时较长，请耐心等待..."):
                    evaluator.preload_models()
        
        with wait_init:
            st.empty()
else:
    st.session_state.messages.clear()
    st.session_state['best_bgr'] = None
    st.session_state['img_bgr'] = None
    # 显式清理 Evaluator 持有的 GPU 资源
    evaluator = st.session_state.get('evaluator')
    if evaluator is not None and hasattr(evaluator, 'cleanup'):
        try:
            evaluator.cleanup()
        except Exception:
            pass
    st.session_state['evaluator'] = None
    load_bgr_img_from_file.clear()
    get_thumbnail_img.clear()
    unload_models()
    extract_funcs.clear()
    get_evaluator.clear()
    get_thumb_evaluator.clear()
    st.stop()

if st.session_state.ui_scene == "Playground":
    render_playground()

if st.session_state.ui_scene != "Chat":
    st.stop()
    # --- 渲染历史聊天记录 ---

if not st.session_state.messages:
    extract_funcs.clear()
    
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        render_message_content(msg, i)

user_feedback = st.chat_input(
    '描述你的增强要求或对上轮结果的反馈\n（例如："这张图有些模糊，给我锐化一下" 或 "这版锐化过度了，稍微柔和一点"）', 
    disabled=not upload or selected_model == DEBUG_MODEL_NAME
)

if upload and selected_model != DEBUG_MODEL_NAME:
    image_analyze()

if user_feedback:
    if not upload:
        st.warning("请先上传图片")
        st.stop()
    
    # 如果选择了模型但没有配置API URL，则警告
    if selected_model != DEBUG_MODEL_NAME and not api_url:
        st.warning("请输入API URL")
        st.stop()
    
    if selected_model == DEBUG_MODEL_NAME:
        pass
    
    # 1. 记录人类用户的输入
    user_feedback_msg = {"role": "user", "content": user_feedback}
    st.session_state.messages.append(user_feedback_msg)
    with st.chat_message(user_feedback_msg["role"]):
        render_message_content(user_feedback_msg, len(st.session_state.messages) - 1)

    # 2. 启动智能体响应
    with st.chat_message("assistant"):
        # 整合上下文策略：将初始目标、历史反馈和当前诉求组装给 LLM

        # 检查是否为无模型测试模式
        if selected_model == DEBUG_MODEL_NAME:
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
                "best_params": best_params,
                "new_tool": None,
                "test_mode": True
            })
            
            st.rerun()

        client, orch = get_orchestrator()

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

            best_queue = deque(maxlen=1)
            prev_img_bgr = get_previous_img(len(st.session_state.messages))
            # ===== [修改] 创建回调时传入自适应早停参数 =====
            callback = StOptunaCallbackImg(
                n_trials, 
                progress_bar, status_text, table_placeholder, 
                best_img, best_queue,
                prev_img_bgr if prev_img_bgr is not None else img_bgr, 
                prev_img_bgr is None, 
                preview_img_max_side, preview_img_scale,
                # ===== [新增] 自适应早停参数 =====
                enable_early_stop=enable_early_stop,
                patience=patience,
                min_trials=min_trials,
                improvement_threshold=improvement_threshold
            )

            best_bgr, best_params, log = None, None, None
            evaluate_code_str = ''
            process_code_str = ''
            new_tool: dict | None = None
            # ===== [新增] 记录实际使用的trial数 =====
            actual_n_trials = n_trials

            evaluate_handler = StStreamResHandler(eva_status, eva_thinking_container)

            input_from_previous = bool(step_by_step and st.session_state['best_bgr'] is not None)
            img_to_process = st.session_state['best_bgr'] if input_from_previous else img_bgr

            for t, body in orch.prepare_stream(
                image=img_to_process,
                model_cache=get_model_cache(),
                device=st.session_state.get('device_learning_evaluator'),
                user_prompt=f"{generate_user_prompt(user_feedback, True, True, step_by_step)}\n\n{runtime_hint}",
                max_side=process_img_max_side if low_res_process else 0
            ):
                if t == "CODE_EVALUATE.START":
                    eva_status.update(state="running")
                elif t == "CODE_EVALUATE.REASONING":
                    evaluate_handler.thinking_chunk(body)
                elif t == "CODE_EVALUATE.STREAM":
                    evaluate_handler.content_chunk(body)
                    evaluate_handler.thinking_end()
                elif t == "CODE_EVALUATE.END":
                    evaluate_handler.content_end()
                elif t == "CODE_EVALUATE.ERROR_RETRY":
                    eva_status.update(label="📝 评价策略验证失败，正在重试", state="error")
                    with eva_thinking_container:
                        st.warning(body)
                elif t == "FATAL_ERROR":
                    eva_status.update(state="error")
                    main_status.update(label="本轮运行失败", state="error")
                    append_run_error_message(body)
                    st.stop()
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
                    image=img_to_process,
                    evaluate_code_str=evaluate_code_str,
                    best_queue=best_queue,
                    user_prompt=f"{generate_user_prompt(user_feedback, True, True, step_by_step)}\n\n{runtime_hint}",
                    n_trials=n_trials,
                    callbacks=[callback],
                    max_side=process_img_max_side if low_res_process else 0
                ):
                    if t == "CODE.START":
                        code_status.update(label="🧠 LLM 调整增强代码", state="running")
                    elif t == "CODE.REASONING":
                        coder_handler.thinking_chunk(body)
                    elif t == "CODE.STREAM":
                        coder_handler.content_chunk(body)
                        coder_handler.thinking_end()
                    elif t == "CODE.END":
                        coder_handler.content_end()
                        process_code_str = body
                    elif t == "CODE.ERROR":
                        code_status.update(label="🧠 增强代码执行失败，正在反馈给 LLM 重试", state="error")
                        with code_thinking_container:
                            st.warning(body)
                    elif t == "OPTUNA.START":
                        optuna_status.update(state="running")
                    elif t == "OPTUNA.END":
                        optuna_status.update(state="complete")
                    elif t == "FATAL_ERROR":
                        code_status.update(state="error")
                        optuna_status.update(state="error")
                        main_status.update(label="本轮运行失败", state="error")
                        append_run_error_message(body)
                        st.stop()
                    elif t == "FINISH":
                        optuna_status.update(state="complete")
                        # ===== [修改] 解包返回值，获取实际trial数 =====
                        best_bgr, best_params, log, actual_n_trials = body

                        coding_finish = True
                        break

                    elif t == "TOOL_REQUEST":
                        code_status.update(state='error')
                        tool_request = body['description']

                        if github_token or huggingface_token or modelscope_token:
                            with main_container:
                                with (tool_status := st.status("⌨️ LLM 编写额外工具", state="error")):
                                    search_container = st.container(border=False)

                                tool_status.update(state='running')
                                if search_container:
                                    allow_learning_process = bool(st.session_state.get("enable_learning_process", False))
                                    process_operator_preference = normalize_process_operator_preference(
                                        st.session_state.get("process_operator_preference"),
                                        allow_learning_process
                                    )
                                    allowed_search_sources = get_allowed_search_sources(
                                        allow_learning_process,
                                        process_operator_preference
                                    )
                                    searcher = Searcher(
                                        client, selected_model,
                                        github_token=github_token,
                                        huggingface_token=huggingface_token,
                                        modelscope_token=modelscope_token,
                                        allowed_sources=allowed_search_sources
                                    )
                                    search_result = StSearch(
                                        searcher, tool_request, search_container, search_steps_limit, search_interval
                                    )
                                    if isinstance(search_result, dict):
                                        if not search_result.get("findings_enriched"):
                                            search_result = StEnrichFindings(searcher, search_result, search_container)
                                        if search_result.get("download_error"):
                                            st.warning(f"模型资产下载失败：{search_result['download_error']}")
                        break
                
                if tool_request:
                    with main_container:
                        if not tool_status:
                            tool_status = st.status("⌨️ LLM 编写额外工具", state="error")

                        with tool_status:
                            with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
                                toolmaker_container = st.container(border=False)
                                toolmaker_placeholder = st.empty()

                    toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
                        
                    runtime_imports = (
                        search_result.get("additional_imports")
                        if isinstance(search_result, dict) else None
                    )
                    runtime_packages = (
                        search_result.get("additional_packages")
                        if isinstance(search_result, dict) else None
                    )

                    for t, body in orch.toolmaker_stream(
                        tool_request,
                        search_result,
                        additional_imports=runtime_imports,
                        additional_packages=runtime_packages,
                        runtime_context=toolmaker_runtime_hint
                    ):
                        if t == "CODE_TOOL.STREAM":
                            toolmaker_handler.content_chunk(body)
                            toolmaker_handler.thinking_end()
                        elif t == "CODE_TOOL.REASONING":
                            toolmaker_handler.thinking_chunk(body)
                        elif t == "CODE_TOOL.TEST":
                            with toolmaker_placeholder:
                                st.info("正在测试算子性能, 请稍等...")
                        elif t == "CODE_TOOL.END":
                            with toolmaker_placeholder:
                                st.info("完成！")
                            toolmaker_handler.content_end()
                        elif t == "ERROR_RETRY":
                            with toolmaker_placeholder:
                                st.warning(body)
                            with main_container:
                                with tool_status:
                                    with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
                                        toolmaker_container = st.container(border=False)
                            
                            toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
                        elif t == "FATAL_ERROR":
                            tool_status.update(state="error")
                            main_status.update(label="本轮运行失败", state="error")
                            append_run_error_message(body)
                            st.stop()
                        elif t == "FINISH":
                            body["additional_imports"] = runtime_imports or []
                            body["additional_packages"] = runtime_packages or []
                            new_tool = body

        # --- 收尾与状态更新 ---
        if best_bgr is None:
            main_status.update(label="本轮运行失败", state="error")
            append_run_error_message("此轮运行失败：未得到可用增强结果。请调整要求后重试。")
            st.stop()
        else:
            # ===== [新增] 根据实际trial数显示不同消息 =====
            if actual_n_trials == 0:
                main_status.update(
                    label=f"没有需要调优的参数", 
                    state="complete"
                )
            elif actual_n_trials < n_trials:
                main_status.update(
                    label=f"本轮调整结束（实际运行 {actual_n_trials}/{n_trials} 轮，已提前收敛）", 
                    state="complete"
                )
            else:
                main_status.update(label="本轮调整结束", state="complete")
            
            st.session_state['best_bgr'] = best_bgr

            new_msg = {
                "role": "assistant",
                # ===== [修改] 显示实际运行的trial数 =====
                "content": f"已完成本轮调优（实际运行 {actual_n_trials} 轮）。请查看图像，如果需要进一步调整（如：增加亮度、减少对比度），请直接告诉我。",
                "image": best_bgr,
                "eval_code": evaluate_code_str,
                "process_code": process_code_str,
                "best_params": best_params,
                "input_from_previous": input_from_previous,
                "input_source": "previous_result" if input_from_previous else "original",
                # ===== [新增] 保存实际trial数到消息历史 =====
                "n_trials_used": actual_n_trials,
            }
            if new_tool: new_msg["new_tool"] = new_tool
            st.session_state.messages.append(new_msg)

            render_message_content(new_msg, len(st.session_state.messages) - 1)
