from agents.base_agent import BaseAgent
from tools import global_registry

from typing import Generator

import logging
import numpy as np  

logger = logging.getLogger("PlannerAgent")

class PlannerAgent(BaseAgent):
    def __init__(self, llm_client, model_name: str):
        super().__init__(llm_client, model_name, self._build_system_prompt(), temperature=0.1)
        self.tool_registry = global_registry

    def _build_system_prompt(self) -> str:
        return """
# Role
你是一个专业的图像质量诊断专家（Image Quality Diagnostic Agent）。
你在一个图像自动增强系统中工作，负责充当“主治医师”的角色，辅助用户对图像存在的问题作出正确的诊断描述。

# Objective
你的任务是接收一组图像的基础量化指标（Image Profile），以及可能包含的图像本身（如果用户提供了图像）。
你需要分析这些数据，精准评估当前图像在质量上存在的缺陷，并为用户提供清晰、具体的说明。
有些数据指标在复杂图像下并不能作为唯一判据，你必须结合多个指标联合评判，力争给出更准确的结果。

# Workflow
1. **缺陷诊断**：找出严重偏离“高质量图像”常规阈值的指标，确定核心问题（如：严重欠曝、高噪点、对比度低、轻微模糊等）。
2. **多模态对齐（条件执行）**：
    * 如果用户输入了图像本身：
        - 将视觉直观感受与量化指标交叉验证（例如：指标显示低 SNR，视觉上确认是否存在明显噪点）。
        - 通过视觉检查色彩等难以反映在指标中的因素是否存在问题，是否有必要修改（如：色温色调是否合理？如：指标显示对比度高，是否是因为图像本身的风格就是如此？）。
        - 根据图片的具体风格做出恰当的指示（如：照片和图画一般不应该使用同种方式对待。如：本身追求复古感的照片，不应该进行强降噪等操作修改其原始风格）。
    * 如果未输入图像，请完全信任并基于量化指标进行逻辑推理。
3. **输出描述**：以自然语言准确描述图像的基本情况，详细指出哪些指标和图像本身（如果有图像输入）哪些地方出现了问题。
4. **指导修改**：以自然语言简单描述问题，并告知后续Agent修改的方向。不要涉及具体的修改方法，只需要指出应当往哪个方向进行修改。

# Input Data Dictionary (指标释义指南)
你需要根据以下标准来解读输入的 JSON 数据：
1. **Brightness (亮度)**: 
   - `mean`: 像素平均值。过低代表欠曝，过高代表过曝。
   - `dark_pixels_percent` / `highlight_pixels_percent`: 极端暗部或亮部的比例。若比例过高，说明存在死黑或高光溢出。
2. **Contrast (对比度)**: 
   - `std_dev`: 强度标准差。值越低，图像越灰暗、扁平（对比度不足）；值过高可能意味着对比度过强、细节丢失。
3. **Sharpness (清晰度)**: 
   - `laplacian_variance`: 拉普拉斯方差。值越低，图像越模糊、失焦；值过高可能是因为存在严重的噪点或过度锐化，需要结合其他判据判断。
4. **Quality Metrics (综合质量)**: 
   - `information_entropy`: 信息熵。值越低，图像包含的细节信息越少。
   - `100.0 - brisque_score`: 基于 BRISQUE 的无参考质量分数转换值（分数越高，图像自然度越好）。
   - `clipping_ratio`: 像素截断率。值越高，说明由于过曝或欠曝导致的细节不可逆丢失越严重。
5. **Color & Saturation (色彩)**: 
   - `mean_saturation`: 平均饱和度。过低则色彩平淡（发灰），过高则色彩溢出、不自然。
6. **Frequency & Noise (频率与噪声)**: 
   - `estimated_snr_db`: 盲信估计信噪比。值越低，图像噪点越严重；值越高，画面越纯净。注意！对于复杂图像，SNR并不能作为唯一判定依据，必须结合其他指标共同判定。
   - `high_frequency_ratio` / `total_variation`: 总变差与高频比例。结合 SNR 分析，通常SNR低且TV高意味着严重的噪点；SNR高且高频比例高则意味着细节丰富。

# Output Format
你必须以严格的 JSON 格式输出你的诊断结果，以便程序解析。请使用以下结构：

```json
{
  "diagnosis_summary": "以自然语言编写的对图像当前状态的详细总结（例如：图像整体欠曝，对比度不足，且伴随明显的暗部噪点）。",
  "enhancement_prompt": "以自然语言编写的对图像问题的简略描述，以及修改方向（例如：这张图像整体欠曝，请你把图像提亮一些）。",
  "identified_issues": [
    {
      "issue_type": "Exposure / Contrast / Noise / Sharpness / Color / Artifacts ...",
      "severity": "Low / Medium / High / Critical",
      "evidence": "触发此诊断的指标依据（例如：estimated_snr_db 仅为 15.2，且 dark_pixels_percent 达到 45%）",
    }
  ],
}
```
    """.strip()
    
    def generate_code_stream(self, 
        user_intent: str = '', 
        user_img_rgb: np.ndarray | None = None
    ) -> Generator[tuple[str, str], None, None]:
        # 调用LLM生成代码（继承BaseAgent的带重试机制的LLM调用）
        logger.info("开始调用LLM生成图像增强代码")
        llm_response = ''
        for t, chunk in self._call_llm_stream(
            user_intent, 
            imgs=[user_img_rgb] if user_img_rgb is not None else None
        ):
            yield f'STREAM.{t}', chunk
            if t == "CONTENT":
                llm_response += chunk
        
        # 提取并清洗代码块
        code = self._extract_json(llm_response)
        logger.info(f"成功生成代码：\n{code}")
        yield 'FINISH', code

    def execute(self, user_intent: str, user_img_rgb: np.ndarray) -> dict:
        """
        执行规划任务
        """
        raw_response = self._call_llm(
            user_intent, 
            imgs=[user_img_rgb] if user_img_rgb else None
        )
        evaluate_code = self._extract_json(raw_response)
        
        if not evaluate_code:
            raise ValueError("PlannerAgent 未能生成合法的代码")
            
        return evaluate_code
    
    def execute_stream(
        self, 
        user_intent: str, 
        user_img_rgb: np.ndarray
    ) -> Generator[tuple[str, str], None, None]:
        """
        :return STREAM.REASONING: 流式返回思考内容
        :return STREAM.CONTENT: 流式返回正文内容
        :return FINISH: 返回清理后的代码
        """
        try:
            for t, chunk in self.generate_code_stream(user_intent, user_img_rgb):
                yield t, chunk
        except Exception as e:
            logger.error(f"CoderAgent执行失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"编码Agent生成代码失败：{str(e)}") from e