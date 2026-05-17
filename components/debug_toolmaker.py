import streamlit as st

from core.searcher import Searcher
from core.orchestrator import Orchestrator
from utils import get_openai_client

from components.tool_search import StEnrichFindings, StSearch, StStreamResHandler
from components import render_tool_save_button

def render_toolmaker(orch: Orchestrator):
    tool_request = st.chat_input("工具要求")
    if not tool_request:
        return
    
    client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
    
    main_container = st.container(border=True)
    search_result = dict()
    if st.session_state.github_token or st.session_state.huggingface_token or st.session_state.modelscope_token:
        with main_container:
            with (tool_status := st.status("⌨️ LLM 编写额外工具", state="error")):
                search_container = st.container(border=False)

            tool_status.update(state='running')
            if search_container:
                allow_learning_process = bool(st.session_state.get("enable_learning_process", False))
                process_operator_preference = str(
                    st.session_state.get("process_operator_preference", "traditional_only")
                )
                allowed_search_sources = (
                    ("github", "huggingface", "modelscope")
                    if allow_learning_process and process_operator_preference != "traditional_only" else
                    ("github",)
                )
                searcher = Searcher(
                    client, st.session_state.selected_model,
                    github_token=st.session_state.github_token,
                    huggingface_token=st.session_state.huggingface_token,
                    modelscope_token=st.session_state.modelscope_token,
                    allowed_sources=allowed_search_sources
                )
                search_result = StSearch(
                    searcher, tool_request, search_container, 
                    st.session_state.search_steps_limit, 
                    st.session_state.search_interval
                )
                if isinstance(search_result, dict):
                    search_result = StEnrichFindings(searcher, search_result, search_container)
                    if search_result.get("download_error"):
                        st.warning(f"模型资产下载失败：{search_result['download_error']}")

    st.json(search_result)

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
            additional_packages=runtime_packages
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
                    st.error(body or "测试失败，要求重新编写")
                with main_container:
                    with tool_status:
                        with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
                            toolmaker_container = st.container(border=False)
                            toolmaker_placeholder = st.empty()
                
                toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
            elif t == "FATAL_ERROR":
                with toolmaker_placeholder:
                    st.error(body or "工具生成失败，已停止。")
                tool_status.update(state="error")
                break
            elif t == "FINISH":
                body["additional_imports"] = runtime_imports or []
                body["additional_packages"] = runtime_packages or []
                with main_container:
                    st.info("运行结束")
                    st.markdown('```python\n' + body.get('code') + '\n```')
                    st.json(body['schema'])
                    render_tool_save_button(
                        body,
                        button_label="🆕 保存新工具",
                        button_key=f"save_tool_debug_{body.get('schema', {}).get('name', 'unknown')}"
                    )
                new_tool = body
