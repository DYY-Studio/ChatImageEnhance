import streamlit as st

from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.delta_generator import DeltaGenerator

from core.searcher import Searcher

def StSearch(
    searcher: Searcher, tool_request: str, container: DeltaGenerator, 
    steps_limt: int = 30, 
    interval: float = 0.5
):
    with container:
        status = st.status("🌐 搜索 GitHub", state="error")
        thought = dict()
        status.update(state="running")
        with status:
            for t, body in searcher.search(tool_request, steps_limt, interval):
                if t.startswith("THINK"):
                    with st.chat_message("ai"):
                        if (value := thought.get(f"SEARCH.REASONING.{t.split('.')[-1]}")):
                            with st.expander("显示思考"):
                                st.write(value)
                        st.write(body)
                elif t.startswith("SEARCH.REASONING"):
                    if not t in thought:
                        thought[t] = ''
                    thought[t] += body
                elif t in ('SEARCH.API_LIMIT_REACHED', 'SEARCH.STEPS_LIMIT_REACHED'):
                    st.write('已达到 GitHub Search API 访问上限')
                    status.update(state="error")
                    return None
                elif t == "SEARCH.FINISH":
                    status.update(state="complete")
                    return body