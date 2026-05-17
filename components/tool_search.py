import streamlit as st

from streamlit.delta_generator import DeltaGenerator
from components.llm_response_handler import StStreamResHandler

from core.searcher import Searcher

def _format_bytes(size: int | None) -> str:
    if size is None:
        return "未知大小"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def StEnrichFindings(
    searcher: Searcher,
    findings: dict | None,
    container: DeltaGenerator | None = None
) -> dict:
    if not isinstance(findings, dict):
        return {}

    target = container or st.container()
    progress_slots: dict[str, DeltaGenerator] = {}

    with target:
        status = st.status("📦 准备模型资产", state="running")
        with status:
            note = st.empty()
            note.write("正在解析依赖与模型资产...")

            def on_progress(event: dict):
                event_name = event.get("event")
                filename = str(event.get("filename") or "asset.bin")
                index = event.get("index") or 1
                total = event.get("total") or 1
                downloaded_bytes = event.get("downloaded_bytes")
                total_bytes = event.get("total_bytes")
                key = f"{index}:{filename}"

                if event_name == "asset_start":
                    st.write(f"开始下载 {index}/{total}: `{filename}`")
                    progress_slots[key] = st.progress(
                        0,
                        text=f"{filename}: 正在连接..."
                    )
                    return

                if event_name == "asset_skip":
                    st.write(
                        f"已存在，跳过下载 {index}/{total}: "
                        f"`{filename}` ({_format_bytes(downloaded_bytes)})"
                    )
                    progress_slots[key] = st.progress(
                        1.0,
                        text=f"{filename}: 已存在"
                    )
                    return

                if event_name == "asset_progress":
                    slot = progress_slots.get(key)
                    if slot is None:
                        slot = st.progress(0, text=f"{filename}: 正在下载...")
                        progress_slots[key] = slot

                    if total_bytes:
                        ratio = min(float(downloaded_bytes or 0) / float(total_bytes), 1.0)
                        slot.progress(
                            ratio,
                            text=(
                                f"{filename}: {_format_bytes(downloaded_bytes)} / "
                                f"{_format_bytes(total_bytes)}"
                            )
                        )
                    else:
                        slot.progress(
                            0,
                            text=f"{filename}: 已下载 {_format_bytes(downloaded_bytes)}"
                        )
                    return

                if event_name == "asset_done":
                    slot = progress_slots.get(key)
                    if slot is None:
                        slot = st.progress(0)
                        progress_slots[key] = slot
                    slot.progress(
                        1.0,
                        text=f"{filename}: 下载完成 ({_format_bytes(downloaded_bytes)})"
                    )

            enriched = searcher.enrich_findings(
                findings,
                auto_download=True,
                progress_callback=on_progress
            )

            if enriched.get("download_error"):
                status.update(label="📦 模型资产准备失败", state="error")
            else:
                status.update(label="📦 模型资产准备完成", state="complete")
            return enriched


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
                elif t == 'SEARCH.STEPS_LIMIT_REACHED':
                    with chat_message:
                        st.write('🚫 已达到搜索步骤数上限')
                        status.update(state="error")
                        return None
                elif t == 'SEARCH.API_LIMIT_REACHED':
                    with chat_message:
                        st.write('⚠️ GitHub Search API 访问上限已达，系统将继续尝试 HuggingFace / ModelScope')
                elif t == "SEARCH.FINISH":
                    status.update(state="complete")
                    with chat_message:
                        st.write("结果已提交")
                    return body
                
                elif t == "TOOL_CALL":
                    with chat_message:
                        with st.expander("显示工具调用结果"):
                            st.table(
                                body,
                                border="horizontal",
                            )
