# 示范代码，本次框架设计并不会使用到Planner
from agents.base_agent import BaseAgent
from tools import global_registry

class PlannerAgent(BaseAgent):
    def __init__(self, llm_client, model_name: str):
        system_prompt = (
            "你是一个资深的计算机视觉架构师。\n"
            "你的任务是根据用户需求，从可用算子库中挑选合适的算子，组成一个图像处理管线(Pipeline)。\n"
            "你必须严格返回合法的 JSON 格式，不要包含任何其他解释文字。"
        )
        super().__init__(llm_client, model_name, system_prompt, temperature=0.1)
        self.tool_registry = global_registry

    def execute(self, user_intent: str) -> dict:
        """
        执行规划任务
        """
        # 1. 获取当前系统注册了哪些可用的 OpenCV 算子
        tools_schema = self.tool_registry.get_all_schemas_for_llm()
        
        # 2. 拼接 User Prompt
        prompt = f"""
        用户需求: "{user_intent}"
        
        当前可用算子库:
        {tools_schema}
        
        请输出 JSON 格式的执行计划。示例格式：
        {{
            "pipeline": ["safe_denoise_bilateral", "safe_enhance_clahe"],
            "reasoning": "根据用户想提取发黄文字的需求，先去噪再增强局部对比度"
        }}
        """
        
        # 3. 调用基类方法进行通信和容错
        raw_response = self._call_llm(prompt)
        
        # 4. 调用基类方法提取干净的数据结构
        plan_json = self._extract_json(raw_response)
        
        if not plan_json:
            raise ValueError("Planner Agent 未能生成合法的计划。")
            
        return plan_json