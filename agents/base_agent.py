import json
import re
import time
import logging
import base64
import io
import numpy as np
import openai

from PIL import Image
from typing import Any, Optional, Generator

# 配置简单的日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseAgent:
    """
    所有大模型 Agent 的基类。

    负责：LLM 通信接口、容错重试、JSON 清洗与提取。
    """
    def __init__(self, llm_client: openai.OpenAI, model_name: str, system_prompt: str, temperature: float = 0.2):
        """
        初始化 Agent

        :param llm_client: 注入的大模型客户端实例 (如 openai.Client)
        :param model_name: 使用的模型名称 (如 `gpt-4o-mini` 或本地 `qwen2.5-coder`)
        :param system_prompt: 该 Agent 的系统提示词
        :param temperature: 逻辑任务建议保持低 temperature (0.0 - 0.2)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

    def _generate_messages(self, user_prompt: str, imgs: list[np.ndarray | Image.Image] | None = None):
        user_content = []

        # 处理图像输入
        if imgs:
            for img in imgs:
                img_process = None
                if isinstance(img, np.ndarray):
                    img_process = Image.fromarray(img)
                elif isinstance(img, Image.Image):
                    img_process = img
                else:
                    raise ValueError("Unsupported Image Type")
                
                with io.BytesIO() as imgbuf:
                    img_process.save(imgbuf, 'png', compression = 1)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(imgbuf.getbuffer()).decode()}"
                        }
                    })
                
        if user_prompt:
            user_content.append({
                "type": "text", 
                "text": user_prompt
            })
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(self, 
        user_prompt: str = '', 
        max_retries: int = 3, 
        imgs: list[np.ndarray | Image.Image] | None = None
    ) -> str:
        """
        底层 LLM 调用方法，包含网络错误重试机制。

        【注】：这里以 OpenAI API 格式为例。

        如果计划流式输出显示，可能需要更改，与前端沟通一下如何实现
        """

        for attempt in range(max_retries):
            try:
                # 这里假设使用的是 OpenAI 或兼容的 API SDK
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=self._generate_messages(user_prompt, imgs),
                    temperature=self.temperature,
                    # response_format={ "type": "json_object" } # 如果模型支持强制 JSON 模式，可以开启
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logging.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避重试 (1s, 2s, 4s...)
                else:
                    raise RuntimeError(f"LLM 调用最终失败，请检查网络或模型服务: {str(e)}")
                
    def _call_llm_stream(self, 
        user_prompt: str = '', 
        max_retries: int = 3, 
        imgs: list[np.ndarray | Image.Image] | None = None
    ) -> Generator[tuple[str, str], None, None]:
        """
        底层 LLM 调用方法，包含网络错误重试机制。

        该函数为流式函数，返回的信令为

        :param REASONING: 表示该块为思考内容
        :param CONTENT: 表示该块为结果内容
        """
        for attempt in range(max_retries):
            try:
                # 这里假设使用的是 OpenAI 或兼容的 API SDK
                stream = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=self._generate_messages(user_prompt, imgs),
                    temperature=self.temperature,
                    # response_format={ "type": "json_object" } # 如果模型支持强制 JSON 模式，可以开启
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        yield "REASONING", delta.reasoning_content
                        continue
                    if hasattr(delta, 'content') and delta.content:
                        yield "CONTENT", delta.content
                break
                
            except Exception as e:
                logging.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避重试 (1s, 2s, 4s...)
                else:
                    raise RuntimeError(f"LLM 调用最终失败，请检查网络或模型服务: {str(e)}")

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        """
        （占位）极其重要的防御性方法：从大模型的回复中强行挖出 JSON。

        对付模型在 JSON 前后说废话的毛病。
        """
        try:
            # 1. 尝试直接解析 (万一模型很乖，直接输出了纯 JSON)
            return json.loads(text)
        except json.JSONDecodeError:
            # 2. 用正则提取 Markdown 代码块中的 JSON: ```json ... ``` 
            # 或直接提取花括号 {...} 或方括号 [...] 的内容
            match = re.search(r'```json\)?\s*([\s\S]*?)\s*```', text)
            if not match:
                match = re.search(r'({[\s\S]*})', text) # 兜底提取花括号
                
            if match:
                json_str = match.group(1).strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logging.error(f"提取出的 JSON 字符串仍存在语法错误: {e}\n原文:\n{json_str}")
                    return None
            else:
                logging.error(f"无法在 LLM 回复中找到合法的 JSON 结构。\n回复原文:\n{text}")
                return None

    def execute(self, **kwargs) -> Any:
        """
        抽象方法，子类必须实现。

        子类在这里拼装特定的 prompt，调用 _call_llm
        """
        raise NotImplementedError("子类必须实现 execute 方法。")
    
    def execute_stream(self, **kwargs) -> Generator[tuple[str, Any], None, None]:
        """
        抽象方法，可选实现

        子类在这里拼装特定的 prompt，调用_call_llm_stream，以Generator[tuple[str, Any]]形式上报

        结果遵循信令形式，下列两个信令必须实现：

        :param stream: 表示正在传输过程中，这是流的一部分，调用者可以流式输出并组装
        :param finish: 表示传输完成，并返回拼接好的整个流，或返回None（需要与调用者协调）
        """
        pass