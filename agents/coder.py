from agents.base_agent import BaseAgent
from tools import global_registry
from typing import Generator, Literal

import logging
import re
import json

logger = logging.getLogger("CoderAgent")

class CoderAgent(BaseAgent):
    """
    编码Agent：核心职责是将用户的图像增强意图转化为带Optuna trial占位符的Python执行代码。
    属于外循环（LLM拓扑突变）的核心组件，仅在内循环参数调优遇到瓶颈时被唤醒。
    
    核心输出格式：
    - 固定生成名为 process 的函数，参数为 img (图像数组) 和 trial (Optuna trial对象)
    - 函数内部调用 cv_wrappers 中经过防呆封装的CV算子
    - 使用 trial.suggest_* 方法定义可调参的超参数（如clahe_clip、ksize等）
    - 最终返回处理后的图像数组
    """
    def __init__(self, 
        llm_client, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.1, 
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None,
        low_res: bool = False,
        **kwargs
    ):
        """
        初始化编码Agent，继承BaseAgent的LLM通信能力
        
        :param llm_client: 大模型客户端实例（如openai.Client、通义千问客户端等）
        :param model_name: 使用的大模型名称，默认gpt-4o-mini（兼顾效率和代码生成能力）
        :param temperature: 生成温度，低温度保证代码逻辑稳定性（0.0-0.2为宜）
        """
        self.low_res = low_res
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, self._build_system_prompt(), temperature, reasoning_effort, **kwargs)
        # 加载全局算子注册表（供Prompt注入可用CV算子信息）
        self.tools = global_registry._tools
        
        logger.info("CoderAgent 初始化完成，已加载全局算子注册表")

    def rebuild_system_prompt(self):
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """
        构建专属系统提示词，明确代码生成的硬性规则和格式要求
        核心原则：让LLM生成可直接被Optuna调用、容错性强的process函数
        """
        # 从全局注册表中提取所有CV算子的Schema（供LLM参考可用函数）
        tool_schemas = global_registry.get_all_schemas_for_llm()
      
        prompt = f"""
### Role / 角色
你是一个专家级的计算机视觉工程师和 Python 开发者。
你的任务是根据用户的自然语言需求以及客观图像评价指标，利用提供的算子库 `cv_wrappers` 编写一个可供 Optuna 调优的图像处理函数 `process(img, trial)`。

你拥有一个特殊权限：**工具库扩充请求**。如果现有算子无法满足用户的特殊需求，你可以向系统申请制造新工具。

### Workflow / 决策工作流
1. **解析需求**：理解用户想要达到的视觉效果（如“去噪”、“增强对比度”、“二值化”等）。
2. **审阅算子**：从 `cv_wrappers` 提供的 Schema 中选择合适的算子。
3. **关键决策 (Decision Point)**：
   - **情况 A (工具充足)**：如果通过现有算子的组合能够实现或近似实现用户需求，请直接编写 Python 代码。
   - **情况 B (工具缺失)**：如果用户的需求是某种特殊的风格化、特定的底层算法，且现有算子无论如何组合都无法达到目的，请放弃编写代码，转而输出一个 JSON 格式的“新工具请求”。

### Code Constraints / 代码约束
* **函数签名**：必须严格为 `def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:`。
* **参数采样**：必须使用 `trial` 对象获取参数。如果一个参数完全没有调优的价值，必须使用常数来阻止Optuna调优。
    * 例如：`d = trial.suggest_int("Bilateral_Filter_d", 1, 9)`
    * 例如：`sigma_color = trial.suggest_float("Bilateral_Filter_sigma_color", 10.0, 150.0)`
    * 例如：`ksize_median = trial.suggest_categorical("Median_Denoise_ksize", [3, 5, 7])`
* **库访问**：你只能使用 `np` (numpy), `cv2` (opencv-contrib-python), `optuna`, `skimage` (scikit-image) 以及提供的算子库 `cv_wrappers`。
* **算子调用**：所有算子必须通过 `cv_wrappers.算子名(img, **params)` 的形式调用。
* **纯净性**：函数内不要包含 `import` 语句，不要定义全局变量。
* **辅助函数**：允许编写辅助函数简化过程、提高可读性，辅助函数必须嵌套在process函数中。

### Strategy & Best Practices / 策略建议
* **命名规范**：在 `trial.suggest` 中使用 `"{{算子名}}_{{参数名}}"` 的命名方式，防止参数冲突。
* **流程合理性**：遵循经典的 CV 顺序（例如：去噪 -> 增强 -> 边缘检测）。
* **边界保护**：确保每个参数都在 Schema 给定的 `range` 范围内，不要越界。
* **合理约束**：选择符合描述而适当合理的参数范围，减少 Optuna 调优的搜索量，避免使用过于极端的参数导致图像完全不可用。如果一个参数没有调优的价值，必须使用常数来阻止Optuna调优。
* **目的单调**：除非用户的指示本身没有方向性，否则避免使用没有明确指向，在特定区间反复震荡的范围（如[-2.0, 2.0]）。
* **性能考量**：连续使用高开销算子时需要仔细研判，避免流程用时过长。
{
    (
        "* **尺寸相关**: 为了加速优化，该函数将会在缩小尺寸的图像上进行调优，最后应用于高分辨率原图。"
        "因此，严禁使用绝对像素值作为卷积核大小、面积阈值或距离阈值。"
        "你必须使用 trial.suggest_float 获取相对比例系数（如占图像宽度的百分比），并在代码内部动态计算出绝对数值"
        "（例如：area = img.shape[0] * img.shape[1] * trial.suggest_float('area_ratio', ...)）。"
        "对于颜色阈值等与分辨率无关的参数，可以直接优化。"
    ) if self.low_res else ""
}

### Provided Schema / 算子库文档
{tool_schemas}

### Output Format / 输出格式要求
#### 格式 A：输出处理代码 (情况 A)
你必须直接返回代码块，不要包含冗长的解释。代码结构应如下：

``python
def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:
    # 1. 参数采样 (基于 Schema 范围)
    # 2. 图像处理流程
    # 3. 返回处理结果
    ...
```

** 示例 (Few-Shot) **

**User Input:** "我想给一张老照片去噪，但要保留建筑的轮廓，不要变模糊。"
**Assistant Thinking:** 现有算子库中有 Bilateral_Filter，完全可以满足。
**Assistant Response:**
```python
def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:
    # 针对“保边降噪”的需求，选择双边滤波 (Bilateral_Filter)
    
    # 1. 采样参数
    # 用户没有提及噪声的强度，选择较为保守的参数范围以免处理过度
    d = trial.suggest_int("Bilateral_Filter_d", 1, 7)
    s_color = trial.suggest_float("Bilateral_Filter_sigma_color", 10.0, 40.0)
    s_space = trial.suggest_float("Bilateral_Filter_sigma_space", 10.0, 40.0)
    
    # 2. 执行处理
    out = cv_wrappers.Bilateral_Filter(
        img, 
        d=d, 
        sigma_color=s_color, 
        sigma_space=s_space
    )
    
    return out
```

#### 格式 B：请求新工具 (情况 B)
如果你决定请求新工具，必须输出 Markdown 格式的 JSON 块，严格包含以下三个字段：

```json
{{
    "status": "NEED_NEW_TOOL",
    "tool_name": "建议的算子英文名 (如 safe_pencil_sketch)",
    "description": "详细描述该算子需要实现什么功能"
}}
```

** 示例 (Few-Shot) **

**User Input**: "帮我把这张照片处理成极其逼真的复古 CRT 电视机扫描线风"
**Assistant Thinking:** 现有算子主要是基础降噪和色彩调整，缺乏生成扫描线、RGB分离的专用工具，无法完美实现。
**Assistant Response:** 
```json
{{
    "status": "NEED_NEW_TOOL",
    "tool_name": "safe_crt_scanline_effect",
    "description": "需要一个算子来模拟CRT电视效果。增加水平方向的黑色扫描线；产生微小的RGB通道错位(色差)。"
}}
```"""
        return prompt

    def execute_stream(self, user_prompt: str, evaluate_code_str: str = "", error_log: str = ""):
        try:
            response = self._call_llm(user_prompt)
            
            # 1. 检查是否为工具请求 (JSON)
            clean_response = re.sub(r'^```json\n|```$', '', response.strip(), flags=re.MULTILINE).strip()
            try:
                tool_request = json.loads(clean_response)
                if isinstance(tool_request, dict) and tool_request.get("status") == "NEED_NEW_TOOL":
                    yield "TOOL_REQUEST", tool_request
                    return
            except json.JSONDecodeError:
                pass

            # 2. 清洗代码
            clean_code = self._clean_generated_code(response)
            
            # 3. 校验结构 (针对 process 函数)
            if not self._validate_function_structure(clean_code):
                raise RuntimeError("生成的处理函数缺少必要结构或格式不正确")
            
            yield ("FINISH", clean_code)
        except Exception as e:
            logger.error(f"CoderAgent execution failed: {str(e)}")
            yield ("ERROR", str(e))
            raise

    def _clean_generated_code(self, raw_code: str) -> str:
        """执行LLM生成代码的标准化清洗"""
        import re, ast
        # 1. 移除Markdown代码块标记
        cleaned = re.sub(r'^```python\n|```$', '', raw_code, flags=re.MULTILINE).strip()
        
        # 2. 标准化缩进（确保4空格基准缩进）
        lines = [line.rstrip() for line in cleaned.splitlines()]
        if lines and lines[0].startswith(' ' * 4):
            lines = [line[4:] if line.startswith(' ' * 4) else line for line in lines]
        
        # 3. 预校验语法
        try:
            ast.parse('\n'.join(lines))
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {str(e)}")
            # 尝试自动修复缺失的换行符
            if 'unexpected EOF' in str(e):
                lines.append('')
        return '\n'.join(lines)

    def _validate_function_structure(self, code: str) -> bool:
        """验证处理函数结构完整性"""
        import re
        # 必须包含def process(
        if 'def process(' not in code:
            return False
        return True
