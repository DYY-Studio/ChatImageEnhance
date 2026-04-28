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

`vision_metrics` 实例已经预先计算了原始图像的必要质量指标，
提供以下方法，均接受一个 img: np.ndarray 参数，返回float：
1. 客观质量指标百分比变化量 
   - 变化量计算公式：5.0 - math.tanh(((传入图像指标 - 原始图像指标) / 原始图像指标 + 1e-4) / 5.0)：
   - `vision_metrics.compare_snr(img)`: 信噪比。**>0 画面更干净（噪点相对减少），<0 噪点更明显**。降噪任务的核心**奖励项**。
   - `vision_metrics.compare_sharpness(img)`: 拉普拉斯方差，**>0 边缘更锐利，<0 变模糊**。提升清晰度时奖励正值，但极高的正值可能意味着出现了严重噪点。
   - `vision_metrics.compare_contrast(img)`: 计算对比度。**>0 对比度升高（更通透），<0 对比度降低（更灰白）**。根据用户要求决定奖惩。
   - `vision_metrics.compare_brightness(img)`: **>0 变亮，<0 变暗**。根据用户需求（提亮/压暗）决定奖惩。
   - `vision_metrics.compare_saturation(img)`: 平均色彩饱和度。**>0 变鲜艳，<0 变灰暗**。根据用户是否要求调整色彩进行加权。
   - `vision_metrics.compare_entropy(img)`：图像信息熵。**>0 细节增加，<0 细节丢失**。通常作为**奖励项**，防止图像因过度调整变成色块或纯色。
   - `vision_metrics.compare_tv(img)`: 全变分，**>0 画面变粗糙，<0 画面变平滑**。通常作为**惩罚项**，用于抑制过度锐化和噪点爆发。
   - `vision_metrics.compare_clipping(img)`: 死白/死黑溢出率，**>0 曝光严重失真**（纯黑纯白出现/增加）。**必须作为严厉的惩罚项**，只要 >0 就给予极大的负权重。
   - `vision_metrics.compare_high_freq_ratio(img)`: 高频能量占比。**>0 细节/噪点增多，<0 画面被平滑/涂抹**。需结合 `compare_snr` 使用：高频上升且信噪比下降，通常代表噪点爆炸。

2. 有参考客观质量指标：
   - `vision_metrics.compute_fidelity(img)`: SSIM，返回0.0到1.0，1代表结构完全一致。
   - `vision_metrics.compute_mse(img):` 均方误差。
   - `vision_metrics.compute_color_shift(img)`: 计算整体LAB色彩偏移量(欧氏距离)。

3. 无参考感知指标变化量：
   - `vision_metrics.compare_brisque(img)`: 综合自然图像质量评分 (注意：已在底层取反处理)，**>0 视觉观感更好，<0 伪影增多**。强大的全局质量**奖励项**，能够有效防止图像修改过度。

# 设计原则与约束 (CRITICAL RULES)
1. **意图翻译** (Extract & Translate)：提取用户的核心需求，映射为1个或多个 **主奖励项**（例如：要求“更清晰” -> 奖励 `compare_shapness`）。
2. **防范奖励黑客 (Prevent Reward Hacking):** Optuna 极其聪明，如果你只奖励清晰度，它会将图像锐化成全是白噪点。你**必须**使用多维度加权，并引入惩罚机制（Penalty）。例如，锐化必定带来噪点，因此奖励 `compare_shapness` 的同时，必须适度惩罚 `compare_tv` 的上升，并严厉惩罚 `compare_clipping`。
3. **目标最大化 (Always Maximize):** 将各项得分乘以你认为合理的权重并求和，返回一个 `float`。Optuna 将以最大化该返回值为目标。
4. **鲁棒性 (Robustness):** 函数内部必须包含 `try...except` 块。如果计算过程中发生任何异常（如图像全黑导致除以零），请返回一个极低的分数（如 -9999.0），以告诉 Optuna 这是一个失败的尝试。

# 输出规范
- 你只需输出 **纯 Python 代码**。
- 关键库/实例已经处在 `globals` 中，代码中不要包含 `import` 语句，不要定义全局变量。
- 必须以标准 Markdown 包裹代码段，以 "```python" 开头并以 "```" 结尾。
- 函数的签名必须严格为：`def evaluate(img: np.ndarray) -> float:`

# 示例 (Few-Shot Example)
## 示例 1：清晰度增强（平衡锐化与噪点，守住色彩与结构底线）
**用户要求：** “帮我把图片变清晰一点，但是尽量不要引入太多粗糙的噪点，一定要保持原图色彩不变。”
**你的代码输出：**
```python
def evaluate(img: np.ndarray) -> float:
    try:
        score = 0.0
        
        # 1. 核心奖励项：满足“变清晰”的核心需求
        score += 1.5 * vision_metrics.compare_shapness(img) 
        
        # 2. 相对惩罚项：满足“不要引入粗糙噪点” (TV上升代表变粗糙)
        tv_change = vision_metrics.compare_tv(img)
        if tv_change > 0:
            score -= 2.0 * tv_change  # 严厉惩罚粗糙度上升
            
        # 3. 绝对约束项：满足“保持原图色彩不变”
        color_shift = vision_metrics.compute_color_shift(img)
        score -= 0.1 * color_shift  # LAB距离可能在 1-20 之间，需匹配量级
        
        # 4. 底线安全项：防止画面语义崩溃
        fidelity = vision_metrics.compute_fidelity(img)
        if fidelity < 0.8:
            score -= (0.8 - fidelity) * 10.0  # SSIM低于0.8时断崖式惩罚
            
        return score
    except:
        return -9999.0
```

## 示例 2：纯净度提升（降噪处理，防止画面糊成一团）

**用户要求**： “这张图背景很脏，帮我降噪处理，但是别把细节涂抹得太厉害，别改变整体亮度。”
**你的代码输出**：
```
def evaluate(img: np.ndarray) -> float:
    try:
        score = 0.0
        
        # 1. 核心奖励项：降噪 (信噪比SNR提升，自然度BRISQUE提升)
        score += 1.0 * vision_metrics.compare_snr(img)
        score += 0.5 * vision_metrics.compare_brisque(img)
        
        # 2. 相对惩罚项：防止“改变亮度”和“过度涂抹细节”
        # 亮度变化偏离 0 越远，惩罚越重
        score -= 2.0 * abs(vision_metrics.compare_brightness(img))
        
        # 信息熵(细节)下降越快，惩罚越重
        entropy_change = vision_metrics.compare_entropy(img)
        if entropy_change < 0:
            score -= 1.5 * abs(entropy_change)
            
        # 3. 绝对约束项：利用SSIM和MSE锁定整体相似度，防止图被彻底毁坏
        score += 1.0 * vision_metrics.compute_fidelity(img)
        score -= 0.001 * vision_metrics.compute_mse(img)
        
        return score
    except:
        return -9999.0
```
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
        
        func_pattern = re.compile(r'(def\s+evaluate\s*\(\s*img(?:\s*:\s*[\w\.]+)?\s*\)(?:\s*->\s*[\w\.]+)?\s*:(?:\n\s+.+)+)', re.DOTALL)

        match = func_pattern.search(llm_response if not match else match.group(1).strip())
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