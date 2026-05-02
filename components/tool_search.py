import streamlit as st

from streamlit.elements.lib.mutable_status_container import StatusContainer

from core.searcher import Searcher

def StSearch(searcher: Searcher, tool_request: str, status: StatusContainer):
    with status:
        thought = dict()
        status.update(state="running")
        for t, body in searcher.search(tool_request):
            if t.startswith("THINK"):
                with st.chat_message("assistant"):
                    if (value := thought.get(f"SEARCH.REASONING.{t.split('.')[-1]}")):
                        with st.expander("显示思考"):
                            st.write(value)
                    st.write(body)
            elif t.startswith("SEARCH.REASONING"):
                if not t in thought:
                    thought[t] = ''
                thought[t] += body
            elif t == "SEARCH.FINISH":
                return body