import streamlit as st

from streamlit.delta_generator import DeltaGenerator
from components.llm_response_handler import StStreamResHandler

from core.searcher import Searcher

def StSearch(
    searcher: Searcher, tool_request: str, container: DeltaGenerator, 
    steps_limt: int = 30, 
    interval: float = 0.5
):
    with container:
        status = st.status("🌐 网络搜索", state="error")
        status.update(state="running")
        with status:
            chat_message = st.chat_message("assistant")
            handler = StStreamResHandler(chat_message, chat_message)

            for t, body in searcher.search(tool_request, steps_limt, interval):
                if t.startswith("THINK"):
                    handler.set_content(body)
                    chat_message = st.chat_message("assistant")
                    handler = StStreamResHandler(chat_message, chat_message)
                elif t.startswith("SEARCH.REASONING"):
                    handler.thinking_chunk(body)
                elif t.startswith("SEARCH.CONTENT"):
                    handler.thinking_end()
                    handler.content_chunk(body)
                elif t in ('SEARCH.API_LIMIT_REACHED', 'SEARCH.STEPS_LIMIT_REACHED'):
                    if t == 'SEARCH.STEPS_LIMIT_REACHED':
                        st.write('🚫 已达到搜索步骤数上限')
                    elif t == 'SEARCH.API_LIMIT_REACHED':
                        st.write('🚫 已达到 GitHub Search API 访问上限')
                    status.update(state="error")
                    return None
                elif t == "SEARCH.FINISH":
                    status.update(state="complete")
                    st.write("结果已提交")
                    return body
                
                elif t == "TOOL_CALL":
                    with chat_message:
                        with st.expander("显示工具调用结果"):
                            st.table(
                                body,
                                border="horizontal",
                            )