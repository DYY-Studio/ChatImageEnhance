import streamlit as st
import time

from streamlit.delta_generator import DeltaGenerator

from tools import global_registry
from utils import get_thumbnail_img_nocache, get_thumbnail_img, get_thumbnail_size
from components.image_comparison import image_comparison

def render_playground(container: DeltaGenerator | None = None):
    if container is None:
        container = st.container()

    with container:
        tool = st.selectbox("选择算子", global_registry.tools.keys())
        tool_info = global_registry.tools.get(tool)
        schema = tool_info['schema']
        info_col, param_col = st.columns([1, 2])
        with info_col:
            with st.container(border=True, height="stretch"):
                st.subheader("信息")
                st.markdown(schema['description'])
                if 'cost' in schema:
                    st.markdown(f"运算速度: **{schema['cost'].upper()}**")
                st.markdown(f"内置算子: **{'否' if tool_info.get('is_dynamic', False) else '是'}**")
                st.toggle("启用", value=not tool_info.get('is_disabled', False))

        params = schema['parameters']
        param_names = list(params.keys())
        with param_col:
            with st.container(border=True, height="stretch"):
                st.subheader("参数")
                st.table({
                    "名称": param_names,
                    "取值": [
                        (
                            f"range: {params[param_name]['range']}"
                            if 'range' in params[param_name]
                            else f"options: {params[param_name].get('options', '获取失败')}"
                        ) 
                        for param_name in param_names
                    ],
                    "介绍": [params[param_name]['description']  for param_name in param_names]
                })

        with st.container(border=True):
            st.subheader("试用")
            params_try = dict()
            param_tune_col, render_col = st.columns([1, 2])
            for name, param in params.items():
                with param_tune_col:
                    with st.container(border=True):
                        if 'range' in param:
                            params_try[name] = st.slider(
                                name, min_value=param['range'][0], max_value=param['range'][-1],
                                help=param.get("description")
                            )
                        elif 'options' in param:
                            params_try[name] = st.selectbox(
                                name, param['options'],
                                help=param.get("description")
                            )
            with param_tune_col:
                do_try = st.toggle("自动尝试", disabled=st.session_state['img_bgr'] is None)
                if not do_try:
                    do_try = st.button("🔥 尝试一下", disabled=st.session_state['img_bgr'] is None)
            with render_col:
                if do_try:
                    try:
                        start_time = time.perf_counter_ns()
                        applied_img = tool_info['func'](st.session_state['img_bgr'], **params_try)
                        end_time = time.perf_counter_ns()
                        st.markdown(f":small[Process Time: {(end_time - start_time) / 1_000_000:.4f} ms]", text_alignment="center")
                        image_comparison(
                            get_thumbnail_img(
                                st.session_state['img_bgr'],
                                mode="b64",
                                max_side=st.session_state.preview_img_max_side,
                                interpolation=st.session_state.preview_img_scale
                            ),
                            get_thumbnail_img_nocache(
                                applied_img,
                                mode="b64",
                                max_side=st.session_state.preview_img_max_side,
                                interpolation=st.session_state.preview_img_scale
                            ),
                            get_thumbnail_size(st.session_state['img_bgr'], st.session_state.preview_img_max_side)[1],
                            "原图",
                            "套用后"
                        )
                    except Exception as e:
                        st.error(str(e))