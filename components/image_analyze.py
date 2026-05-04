import streamlit as st
import cv2

from components import render_message_content
from agents.planner import PlannerAgent
from utils import get_openai_client

def image_analyze():
    if not st.session_state['messages']:
        start_analyze = False
        if st.button("💡 不知如何描述？让 AI 分析", key="ai_planner_btn", width='stretch', disabled=start_analyze):
            start_analyze = True
            # 1. 模拟用户发起了分析请求
            st.session_state.messages.append({"role": "user", "content": "请帮我分析这张图像的问题，并给出增强建议。"})
            with st.chat_message("user"):
                st.markdown("请帮我分析这张图像的问题，并给出增强建议。")

            # 2. 立即触发 AI 分析流
            with st.chat_message("assistant"):
                with st.status("🔎 AI 正在分析图像客观指标与视觉问题...", expanded=True) as plan_status:
                    client = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url)
                    planner = PlannerAgent(client, st.session_state['selected_model'], reasoning_effort=st.session_state.reasoning_effort)
                    img_rgb = cv2.cvtColor(st.session_state['img_bgr'], cv2.COLOR_BGR2RGB)
                    
                    analyze_result = {}
                    plan_placeholder = st.empty()
                    
                    def plan_stream_wrapper():
                        global analyze_result
                        # 将评价指标一并传给 Planner
                        for t, body in planner.execute_stream(st.session_state['evaluator'].get_profile_yaml(), img_rgb if st.session_state['is_visual_model'] else None):
                            if t in ["STREAM.REASONING", "STREAM.CONTENT"]:
                                yield body
                            elif t == "FINISH":
                                analyze_result = body
                                
                    plan_placeholder.write_stream(plan_stream_wrapper())
                    plan_status.update(label="图像分析完成", state="complete", expanded=False)
                
                # 3. 解析 Planner 的输出并格式化为友好的对话消息
                if analyze_result:
                    summary = analyze_result.get('diagnosis_summary', '未得出明确总结')
                    suggestion = analyze_result.get('enhancement_prompt', '')
                    
                    response_text = f"**📊 图像诊断总结：**\n{summary}\n\n"
                    
                    if 'identified_issues' in analyze_result:
                        response_text += "**🔍 发现的具体问题：**\n"
                        response_text += "| 问题类型 | 严重度 | 依据 |\n"
                        response_text += "| --- | --- | --- |\n"
                        for issue in analyze_result['identified_issues']:
                            response_text += f"| {issue.get('issue_type', '未知')} | {issue.get('severity', '未知')} | {issue.get('evidence')}\n"
                    
                    response_text += f"\n**✨ 推荐增强提示词：**\n```text\n{suggestion}\n```\n"
                    response_text += "\n*💡 您可以直接复制上面的提示词发送给我，或在此基础上做出一定的调整*"
                    
                    # 渲染到界面并存入历史记录
                    new_msg = {"role": "assistant", "content": response_text}
                    st.session_state.messages.append(new_msg)
                    render_message_content(new_msg, len(st.session_state.messages) - 1)