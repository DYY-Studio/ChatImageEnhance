import streamlit as st

from core.searcher import Searcher
from utils import get_openai_client

from components.tool_search import StSearch, StStreamResHandler

def render_toolmaker():
    tool_request = st.chat_input("工具要求")
    if not tool_request:
        return
    
    client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
    
    main_container = st.container(border=True)
    search_result = dict()
    if st.session_state.github_token:
        with main_container:
            with (tool_status := st.status("⌨️ LLM 编写额外工具", state="error")):
                search_container = st.container(border=False)

            tool_status.update(state='running')
            if search_container:
                if st.session_state.github_token:
                    searcher = Searcher(
                        client, st.session_state.selected_model,
                        github_token=st.session_state.github_token,
                        modelscope_token=st.session_state.modelscope_token
                    )
                    search_result: dict = StSearch(
                        searcher, tool_request, search_container, 
                        st.session_state.search_steps_limit, 
                        st.session_state.search_interval
                    )

    st.json(search_result)

    # if tool_request:
    #     with main_container:
    #         if not tool_status:
    #             tool_status = st.status("⌨️ LLM 编写额外工具", state="error")

    #         with tool_status:
    #             with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
    #                 toolmaker_container = st.container(border=False)
    #                 toolmaker_placeholder = st.empty()

    #     toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
            
    #     for t, body in orch.toolmaker_stream(tool_request, search_result):
    #         if t == "CODE_TOOL.STREAM":
    #             toolmaker_handler.content_chunk(body)
    #             toolmaker_handler.thinking_end()
    #         elif t == "CODE_TOOL.REASONING":
    #             toolmaker_handler.thinking_chunk(body)
    #         elif t == "CODE_TOOL.TEST":
    #             with toolmaker_placeholder:
    #                 st.info("正在测试算子性能, 请稍等...")
    #         elif t == "CODE_TOOL.END":
    #             with toolmaker_placeholder:
    #                 st.info("完成！")
    #             toolmaker_handler.content_end()
    #         elif t == "ERROR_RETRY":
    #             with main_container:
    #                 with tool_status:
    #                     with (toolmaker_status := st.status("⌨️ 编写工具", state="error")):
    #                         toolmaker_container = st.container(border=False)
                
    #             toolmaker_handler = StStreamResHandler(toolmaker_status, toolmaker_container)
    #         elif t == "FINISH":
    #             new_tool = body
