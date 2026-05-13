import streamlit as st
import cv2
import numpy as np

from queue import Queue
from streamlit_local_storage import LocalStorage

from core.orchestrator import Orchestrator
from core.evaluator import Evaluator
from core.searcher import Searcher

from agents.coder import CoderAgent
from agents.evaluator import EvaluatorAgent
from agents.toolmaker import ToolMakerAgent

from components.optuna_callbacks import StOptunaCallbackImg
from components.tool_search import StSearch
from components.llm_response_handler import StStreamResHandler
from components.image_analyze import image_analyze
from components.tools_playground import render_playground
from components import get_thumbnail_img_wrapper, render_message_content, get_previous_img, generate_user_prompt

from utils import *
from constants import *

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
if 'reasoning_effort' not in st.session_state:
    st.session_state['reasoning_effort'] = None
if 'process_img_max_side' not in st.session_state:
    st.session_state['process_img_max_side'] = 1500
if 'img_bgr' not in st.session_state:
    st.session_state['img_bgr'] = None
if 'running' not in st.session_state:
    st.session_state['running'] = False
if 'preview_img_max_side' not in st.session_state:
    st.session_state['preview_img_max_side'] = 800
if 'preview_img_scale' not in st.session_state:
    st.session_state['preview_img_scale'] = cv2.INTER_LINEAR

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


with st.sidebar:
    st.header("设置")

    # 全局初始化变量，解决未定义问题
    github_token = st.session_state.github_token

    with st.expander("模式", expanded=True):
        st.segmented_control(
            "执行模式", ["Chat", "Playground", "Deeplearning"],
            required=True, default="Chat", key='ui_scene',
            help="选择Playground以查看并试用所有可用的算子；选择Deeplearning以使用深度学习模式"
        )
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
        proxy_url = st.text_input("HTTP 代理服务器", placeholder="http://localhost:7890", key="proxy_url",
                                  on_change=get_openai_client.clear)
        fetch_button = st.button("获取模型列表", disabled=True if not api_url else False, width="stretch")
        if fetch_button:
            get_models()

        # 添加无模型选项用于测试
        model_options = [DEBUG_MODEL_NAME] + (st.session_state.models or [])
        selected_model = st.selectbox(
            "模型",
            options=model_options,
            index=0,
            key="selected_model",
            disabled=st.session_state.models is None or len(st.session_state.models) == 0
        )

        # 只有非Deeplearning模式显示视觉输入和高级选项
        if st.session_state.ui_scene != "Deeplearning":
            is_visual_model = st.toggle("该模型支持视觉输入", key="is_visual_model")

            with st.expander("高级"):
                reasoning_effort = st.selectbox(
                    "推理努力 Reasoning Effort", ['default', 'minimal', 'low', 'medium', 'high', 'xhigh'],
                    help="请查询你使用的模型是否支持该字段。部分模型不支持minimal和xhigh。"
                )
                if reasoning_effort == "default":
                    reasoning_effort = None
                st.session_state['reasoning_effort'] = reasoning_effort

    # 只有非Deeplearning模式显示预览窗口
    if st.session_state.ui_scene != "Deeplearning":
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

    # 只有非Deeplearning模式显示代码检索窗口
    if st.session_state.ui_scene != "Deeplearning":
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

    # 只有非Deeplearning模式显示处理窗口
    if st.session_state.ui_scene != "Deeplearning":
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

    # 修复：添加冒号 + 变量已全局初始化
    if (cache_api := fetch_button and st.session_state.models) or github_token:
        with st.container(height=1, border=False):
            if cache_api:
                if api_url:
                    localS.setItem("api_url", api_url, "locals_api_url")
                elif localS.getItem("api_url"):
                    localS.deleteItem("api_url", "del_locals_api_url")

                if api_key:
                    localS.setItem("api_key", api_key, "locals_api_key")
                elif localS.getItem("api_key"):
                    localS.deleteItem("api_key", "del_locals_api_key")

                if proxy_url:
                    localS.setItem("proxy_url", proxy_url, "locals_proxy_url")
                elif localS.getItem("proxy_url"):
                    localS.deleteItem("proxy_url", "del_locals_proxy_url")

            if github_token:
                localS.setItem("github_token", github_token, "locals_github_token")
            elif localS.getItem("github_token"):
                localS.deleteItem("github_token", "del_locals_github_token")

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
    return Evaluator(raw_array)


# 如果有历史结果，并在界面顶部展示原图与当前最佳进度的对比
if upload:
    img_bgr = load_bgr_img_from_file(upload)
    st.session_state['img_bgr'] = img_bgr
    img_bgr_preview_bytes = get_thumbnail_img_wrapper(img_bgr, 'binary')

    evaluator = get_evaluator(img_bgr)
    st.session_state['evaluator'] = evaluator
    top_preview_placeholder = st.empty()

    st.subheader("原图")
    st.image(
        img_bgr_preview_bytes, width="stretch",
        caption=f"{img_bgr.shape[1]}x{img_bgr.shape[0]}, {img_bgr.shape[2]} Channel(s)"
    )
    st.divider()
else:
    st.session_state.messages.clear()
    st.session_state['best_bgr'] = None
    st.session_state['img_bgr'] = None
    st.session_state['evaluator'] = None
    load_bgr_img_from_file.clear()
    get_thumbnail_img.clear()
    get_evaluator.clear()
    st.stop()

# 只在Deeplearning模式下且上传图片后显示模型调用和图像增强按钮
if st.session_state.ui_scene == "Deeplearning":
    if upload is not None:
        col1, col2 = st.columns(2)
        with col1:
            model_call_btn = st.button("模型调用", use_container_width=True)
        with col2:
            img_enhance_btn = st.button("图像增强", use_container_width=True)
    else:
        st.info("请先上传图片!")

if st.session_state.ui_scene == "Playground":
    render_playground()

if st.session_state.ui_scene != "Chat":
    # 在Deeplearning模式下，显示按钮但不停止执行
    if st.session_state.ui_scene == "Deeplearning":
        # Deeplearning模式下的特殊处理
        import sys
        import os

        # 添加deeplearning模块路径
        deeplearning_path = os.path.join(os.path.dirname(__file__), 'deeplearning')
        sys.path.insert(0, deeplearning_path)

        # 初始化deeplearning相关的session state变量
        if 'model_called' not in st.session_state:
            st.session_state['model_called'] = False
        if 'enhancement_done' not in st.session_state:
            st.session_state['enhancement_done'] = False

        # 模型调用按钮功能
        if model_call_btn:
            if upload is None:
                st.warning("请先上传图片!")
            else:
                # 创建用于显示LLM输出的容器
                output_placeholder = st.empty()

                # 使用列表包装字符串以在嵌套函数中修改
                output_container = [""]

                def update_output(text):
                    output_container[0] += text
                    # 更新占位符的内容
                    output_placeholder.text_area("LLM 输出日志", value=output_container[0], height=400)

                with st.spinner("正在调用大语言模型生成代码..."):
                    try:
                        # 重新导入模块以确保最新代码生效
                        import sys
                        import importlib
                        if 'deeplearning.controller' in sys.modules:
                            importlib.reload(sys.modules['deeplearning.controller'])

                        from deeplearning.controller import main_with_output as controller_main
                        success = controller_main(update_output)
                        st.session_state['model_called'] = True
                        if success:
                            st.success("模型调用完成！代码已生成。")
                        else:
                            st.warning("模型调用完成，但可能存在一些问题，请检查代码。")
                    except Exception as e:
                        st.error(f"模型调用失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

        # 图像增强按钮功能
        if img_enhance_btn:
            with st.spinner("正在增强图像..."):
                try:
                    # 保存上传的图片到输入目录
                    from PIL import Image
                    import numpy as np
                    from deeplearning.prepare import CFG
                    import io

                    # 确保输入目录存在
                    os.makedirs(CFG["input_dir"], exist_ok=True)

                    # 读取上传的图片并保存到输入目录
                    img = Image.open(upload)
                    input_img_path = os.path.join(CFG["input_dir"], upload.name)
                    img.save(input_img_path)

                    # 运行图像增强
                    from deeplearning.infer import main as infer_main
                    # 由于infer.py没有main函数，我们需要模拟其主要逻辑
                    import torch
                    from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
                    from PIL import ImageOps
                    import torch.nn.functional as F

                    # 从infer.py复制主要逻辑
                    device = torch.device(CFG["device"])
                    model_id = "caidas/swin2SR-compressed-sr-x4-48"
                    print(f"Loading model {model_id}...")
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
                    print("Model loaded successfully.")

                    # 处理上传的图片
                    filename = upload.name
                    input_path = os.path.join(CFG["input_dir"], filename)
                    image = Image.open(input_path).convert("RGB")
                    w, h = image.size

                    # 对图像进行Padding以符合模型要求
                    pad_w = (64 - w % 64) % 64
                    pad_h = (64 - h % 64) % 64
                    padded_image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0)

                    # 推理准备
                    inputs = processor(images=padded_image, return_tensors="pt").to(device)

                    # 执行推理
                    with torch.no_grad():
                        outputs = model(**inputs)
                        output = outputs.reconstruction.squeeze(0).cpu().clamp(0, 1)

                        # 后处理与精准剪裁
                        output_img_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        full_output_image = Image.fromarray(output_img_np)

                        scale_factor = 4
                        final_w, final_h = w * scale_factor, h * scale_factor
                        output_image = full_output_image.crop((0, 0, final_w, final_h))

                        # 保存增强后的图像
                        output_path = os.path.join(CFG["output_dir"], filename)
                        output_image.save(output_path)

                    st.session_state['enhancement_done'] = True
                    st.success("图像增强完成！")

                    # 显示原始图像和增强后的图像
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("原始图像")
                        st.image(upload, caption="原始图像", width="stretch")
                    with col2:
                        st.subheader("增强后图像")
                        st.image(output_image, caption="增强后图像 (4x分辨率)", width="stretch")

                except Exception as e:
                    st.error(f"图像增强失败: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.stop()
    # --- 渲染历史聊天记录 ---

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        render_message_content(msg, i)

# 根据UI场景决定是否显示用户输入框
if st.session_state.ui_scene != "Deeplearning":
    user_feedback = st.chat_input(
        '描述你的增强要求或对上轮结果的反馈\n（例如："这张图有些模糊，给我锐化一下" 或 "这版锐化过度了，稍微柔和一点"）',
        disabled=not upload or selected_model == DEBUG_MODEL_NAME
    )
    
    if upload and selected_model != DEBUG_MODEL_NAME:
        image_analyze()
else:
    # 在Deeplearning模式下不显示输入框和AI分析功能
    user_feedback = None

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

        client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
        orch = Orchestrator(
            CoderAgent(client, selected_model, reasoning_effort=st.session_state.reasoning_effort,
                       low_res=low_res_process),
            EvaluatorAgent(client, selected_model, reasoning_effort=st.session_state.reasoning_effort),
            ToolMakerAgent(client, selected_model, reasoning_effort=st.session_state.reasoning_effort)
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
                with preview_tab:
                    best_img = st.empty()
                with data_tab:
                    table_placeholder = st.empty()

            best_queue = Queue()
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

            for t, body in orch.prepare_stream(
                    image=img_bgr,
                    user_prompt=generate_user_prompt(user_feedback, True, True, step_by_step)
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

                img_to_process = st.session_state['best_bgr'] if step_by_step and st.session_state[
                    'best_bgr'] is not None else img_bgr

                orch.coder.rebuild_system_prompt()
                for t, body in orch.process_stream(
                        image=img_to_process,
                        evaluate_code_str=evaluate_code_str,
                        best_queue=best_queue,
                        user_prompt=generate_user_prompt(user_feedback, True, True, step_by_step),
                        n_trials=n_trials,
                        callbacks=[callback],
                        max_side=process_img_max_side if low_res_process else 0
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
                        # ===== [修改] 解包返回值，获取实际trial数 =====
                        best_bgr, best_params, log, actual_n_trials = body

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
                                            searcher, tool_request, search_container, search_steps_limit,
                                            search_interval
                                        )
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

                    for t, body in orch.toolmaker_stream(tool_request, search_result):
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
            # ===== [新增] 根据实际trial数显示不同消息 =====
            if actual_n_trials < n_trials:
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
                # ===== [新增] 保存实际trial数到消息历史 =====
                "n_trials_used": actual_n_trials,
            }
            if new_tool:
                new_msg["new_tool"] = new_tool
            st.session_state.messages.append(new_msg)

            render_message_content(new_msg, len(st.session_state.messages) - 1)