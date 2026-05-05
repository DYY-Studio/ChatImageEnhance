import streamlit as st

from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.delta_generator import DeltaGenerator

class StStreamResHandler:
    def __init__(self, parent_status: StatusContainer | DeltaGenerator, thinking_container: DeltaGenerator):
        self.content_str = ''
        self.thinking_str = ''

        self._parent_status = parent_status
        self._is_parent_status = isinstance(parent_status, StatusContainer)
        self._thinking_container = thinking_container
        self._thinking_status: StatusContainer | None = None

        self._thinking_delta: DeltaGenerator | None = None
        self._content_delta: DeltaGenerator | None = None

    def content_chunk(self, chunk: str):
        self.content_str += chunk

        if not self._content_delta:
            with self._parent_status:
                self._content_delta = st.empty()
                if self._is_parent_status: self._parent_status.update(state="running")

        if self._is_parent_status and self._parent_status._current_state != "running":
            self._parent_status.update(state="running")
        
        self._content_delta.markdown(self.content_str, unsafe_allow_html=True)

    def thinking_chunk(self, chunk: str):
        self.thinking_str += chunk

        if not self._thinking_status:
            with self._parent_status:
                with self._thinking_container:
                    self._thinking_status = st.status("思考中...", state="running")
                    with self._thinking_status:
                        self._thinking_delta = st.empty()
        
        if self._is_parent_status and self._parent_status._current_state != "running":
            self._parent_status.update(state="running")
        if self._thinking_status._current_state != "running":
            self._thinking_status.update(state="running")

        self._thinking_delta.markdown(self.thinking_str, unsafe_allow_html=True)

    def thinking_end(self):
        if self._thinking_status and self._thinking_status._current_state != "complete":
            self._thinking_status.update(label="查看思考", state="complete")

    def content_end(self):
        if self._is_parent_status and self._parent_status._current_state != "complete":
            self._parent_status.update(state="complete")
            
    def set_content(self, content: str):
        if not self._content_delta:
            with self._parent_status:
                self._content_delta = st.empty()
        
        self._content_delta.markdown(content)