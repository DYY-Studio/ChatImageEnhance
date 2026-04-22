from agents.base_agent import BaseAgent
from tools import global_registry

from typing import Generator

import logging
import re

logger = logging.getLogger("EvaluatorAgent")

class EvaluatorAgent(BaseAgent):
    def __init__(self, llm_client, model_name: str):
        super().__init__(llm_client, model_name, self._build_system_prompt(), temperature=0.1)
        self.tool_registry = global_registry

    def _build_system_prompt(self) -> str:
        return """
# 角色设定
你是一个世界顶尖的计算机视觉（CV）算法工程师与图像质量评估专家。
你的任务是将用户对图像处理的自然语言要求，转化为一个精确、健壮且可量化的 Python 评估函数。
该评估函数将被用于 Optuna 超参数优化框架中，Optuna 会通过不断调整图像处理管线的参数，来 **最大化 (Maximize)** 你编写的评估函数的返回值。

# 可用工具库 (Available Tools)
在编写评估函数时，你**只能**使用预封装在 `vision_metrics` 实例中的函数，以及
* `np` (numpy) 库
* `cv2` (opencv-python) 库
* `skimage` (skimage) 库
* `math` 标准库

绝对不要假设存在其他未列出的第三方库，也不要使用其他标准库。

`vision_metrics` 实例包含以下方法：
1. 客观质量指标：
   - `vision_metrics.compute_snr(img: np.ndarray) -> float`: 计算信噪比，值越大噪点越少。
   - `vision_metrics.compute_sharpness(img: np.ndarray) -> float`: 拉普拉斯方差，计算清晰度/边缘锐度，值越大越锐利。
   - `vision_metrics.compute_contrast(img: np.ndarray) -> float`: 计算对比度。
   - `vision_metrics.compute_brightness(img: np.ndarray) -> float`: 计算平均亮度 (0-255)。
   - `vision_metrics.compute_saturation_metrics(img: np.ndarray) -> float`: 计算平均色彩饱和度。
   - `vision_metrics.compute_entropy(img: np.ndarray) -> float`：计算图像信息熵。用于衡量图像信息的丰富度，过暗、过曝或低对比度图像熵值通常较低。
   - `vision_metrics.compute_tv(img: np.ndarray) -> float`: 计算全变分，量化噪点和过度锐化的粗糙感，已经进行尺寸归一化，确保不同分辨率的图像处于同一量级。

   - `vision_metrics.compute_fidelity(img: np.ndarray) -> float`: SSIM，返回0.0到1.0，1代表结构完全一致。
   - `vision_metrics.compute_mse(img: np.ndarray) -> float:` 均方误差。
   - `vision_metrics.compute_color_shift(img: np.ndarray) -> float`: 计算整体LAB色彩偏移量(欧氏距离)。

2. 无参考感知指标：
   - `vision_metrics.compute_brisque(img: np.ndarray) -> float`: 综合自然图像质量评分 (注意：已在底层取反处理，值越大代表人类观感越好，0到100)。

`vision_metrics` 实例包含以下字段：
1. 预计算的原始图像客观质量指标：
   - `vision_metrics.base_sharpness`
   - `vision_metrics.base_tv`
   - `vision_metrics.base_entropy`
   - `vision_metrics.base_clipping`
   - `vision_metrics.base_saturation`
   - `vision_metrics.base_snr`
   - `vision_metrics.base_hf_ratio`
   - `vision_metrics.base_contrast`

2. 保留的原始图像：
   - `vision_metrics.original_img`：原始图像
   - `vision_metrics.gray_original`：预先将原始图像转换为灰度图像

# 设计原则与约束 (CRITICAL RULES)
1. **防范奖励黑客 (Prevent Reward Hacking):** Optuna 极其聪明，如果你只奖励清晰度，它会将图像锐化成全是白噪点。你**必须**使用多维度加权，并引入惩罚机制（Penalty）。例如，在追求高锐度时，必须扣除低 SNR（高噪点）的惩罚分。
2. **目标最大化 (Always Maximize):** 返回的 `float` 分数越高，代表图像越符合用户需求。
3. **鲁棒性 (Robustness):** 函数内部必须包含 `try...except` 块。如果计算过程中发生任何异常（如图像全黑导致除以零），请返回一个极低的分数（如 -9999.0），以告诉 Optuna 这是一个失败的尝试。
4. **归一化考量:** 尽量让各项指标的分数在相似的量级上进行加权相加，防止某一指标绝对数值过大而淹没其他指标。
5. **变化考量:** 对于客观质量指标，可多使用相对原始图像的变化率保证图像修改方向符合要求

# 输出规范
- 你只需输出 **纯 Python 代码**。
- 关键库/实例已经处在 `globals` 中，代码中不要包含 `import` 语句，不要定义全局变量。
- 必须以标准 Markdown 包裹代码段，以 "```python" 并以 "```" 结尾。
- 函数的签名必须严格为：`def evaluate(img: np.ndarray) -> float:`

# 示例 (Few-Shot Example)
用户输入："请帮我去除照片里的噪点，但一定要保留衣服的纹理细节。"
你的输出：
def evaluate(img: np.ndarray) -> float:
    try:
        # 1. 核心目标：提高信噪比 (降噪)
        new_snr = vision_metrics.compute_snr(img)
        snr_improvement = new_snr - vision_metrics.base_snr
        
        # 2. 保真度约束：必须保留纹理细节
        # SSIM 非常适合用来惩罚过度磨皮导致的结构丢失
        ssim_score = vision_metrics.compute_fidelity(img)
        
        # 3. 颜色防偏移约束
        color_shift = vision_metrics.compute_color_shift(img)
        
        # 构建惩罚项
        structure_penalty = 0.0
        # 如果 SSIM 低于 0.85，说明画面结构被严重破坏，给予严厉惩罚
        if ssim_score < 0.85:
            structure_penalty = (0.85 - ssim_score) * 500.0
            
        color_penalty = color_shift * 2.0
            
        # 目标函数：奖励信噪比提升，但以保持结构相似度和颜色一致为前提
        total_score = (snr_improvement * 5.0) - structure_penalty - color_penalty
        
        return float(total_score)
    except Exception:
        return -9999.0
    """.strip()

    def _extract_code_block(self, llm_response: str) -> str:
        """
        从LLM回复中提取纯Python代码块（剥离Markdown格式和多余解释）
        
        :param llm_response: LLM原始回复文本
        :return: 纯Python代码字符串（仅evaluate函数）
        :raises RuntimeError: 无法从回复中提取合法代码块时抛出异常
        """
        # 正则匹配```python ... ```代码块
        code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
        match = code_pattern.search(llm_response)
        
        if not match:
            # 兜底匹配：无markdown格式时直接提取evaluate函数
            func_pattern = re.compile(r'def evaluate\(img.*?\):\s*(.*?)(?=\ndef|\n$)', re.DOTALL)
            match = func_pattern.search(llm_response)
            if not match:
                logger.error(f"无法从LLM回复中提取代码块，原始回复：{llm_response}")
                raise RuntimeError("LLM回复未包含合法的evaluate函数代码块")
        
        code = match.group(1).strip()
        logger.info("成功从LLM回复中提取evaluate函数代码")
        return code
    
    def generate_code_stream(self, 
        user_intent: str = '', 
    ) -> Generator[tuple[str, str], None, None]:
        # 调用LLM生成代码（继承BaseAgent的带重试机制的LLM调用）
        logger.info("开始调用LLM生成图像增强代码")
        llm_response = ''
        for t, chunk in self._call_llm_stream(user_intent):
            yield f'STREAM.{t}', chunk
            if t == "CONTENT":
                llm_response += chunk
        
        # 提取并清洗代码块
        code = self._extract_code_block(llm_response)
        logger.info(f"成功生成代码：\n{code}")
        yield 'FINISH', code

    def generate_code(self, 
        user_intent: str = '', 
    ) -> str:
        
        # 调用LLM生成代码（继承BaseAgent的带重试机制的LLM调用）
        logger.info("开始调用LLM生成图像增强代码")
        llm_response = self._call_llm(user_intent)
        
        # 提取并清洗代码块
        code = self._extract_code_block(llm_response)
        logger.info(f"成功生成代码：\n{code}")
        return code

    def execute(self, user_intent: str) -> dict:
        """
        执行规划任务
        """
        raw_response = self._call_llm(user_intent)
        evaluate_code = self._extract_code_block(raw_response)
        
        if not evaluate_code:
            raise ValueError("EvaluateAgent 未能生成合法的代码")
            
        return evaluate_code
    
    def execute_stream(
        self, 
        user_intent: str, 
    ) -> Generator[tuple[str, str], None, None]:
        """
        :return STREAM.REASONING: 流式返回思考内容
        :return STREAM.CONENT: 流式返回正文内容
        :return FINISH: 返回清理后的代码
        """
        try:
            for t, chunk in self.generate_code_stream(user_intent):
                yield t, chunk
        except Exception as e:
            logger.error(f"CoderAgent执行失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"编码Agent生成代码失败：{str(e)}") from e