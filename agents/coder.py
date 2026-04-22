from agents.base_agent import BaseAgent
from tools import global_registry
from typing import Generator

import logging
import re

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
    def __init__(self, llm_client, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        初始化编码Agent，继承BaseAgent的LLM通信能力
        
        :param llm_client: 大模型客户端实例（如openai.Client、通义千问客户端等）
        :param model_name: 使用的大模型名称，默认gpt-4o-mini（兼顾效率和代码生成能力）
        :param temperature: 生成温度，低温度保证代码逻辑稳定性（0.0-0.2为宜）
        """
        # 构造CoderAgent专属的系统提示词，明确代码生成规则
        system_prompt = self._build_system_prompt()
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, system_prompt, temperature)
        # 加载全局算子注册表（供Prompt注入可用CV算子信息）
        self.tools = global_registry._tools
        logger.info("CoderAgent 初始化完成，已加载全局算子注册表")

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

### Task / 任务目标
1.  **解析需求**：理解用户想要达到的视觉效果（如“去噪”、“增强对比度”、“二值化”等）。
2.  **构建管线**：从 `cv_wrappers` 提供的 Schema 中选择合适的算子。
3.  **定义搜索空间**：使用 `trial.suggest_int`, `trial.suggest_float` 或 `trial.suggest_categorical` 为算子的每个参数定义合理的范围。
4.  **合理约束**：避免使用过于极端的参数导致图像完全不可用。严禁使用没有明确方向，导致反复震荡的模棱两可的范围（如[-2.0, 2.0]）。如果一个参数没有调优的价值，允许使用常数来阻止Optuna调优。
5.  **生成代码**：输出一段符合 Python 语法的完整 `process` 函数代码。

### Code Constraints / 代码约束
* **函数签名**：必须严格为 `def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:`。
* **参数采样**：必须使用 `trial` 对象获取参数。**当且仅当** 一个参数没有调优的价值，允许使用常数来阻止Optuna调优。
    * 例如：`d = trial.suggest_int("Bilateral_Filter_d", 1, 9)`
    * 例如：`sigma_color = trial.suggest_float("Bilateral_Filter_sigma_color", 10.0, 150.0)`
    * 例如：`ksize_median = trial.suggest_categorical("Median_Denoise_ksize", [3, 5, 7])`
* **库访问**：你只能使用 `np` (numpy), `cv2` (OpenCV), `optuna` 以及提供的算子库 `cv_wrappers`。
* **算子调用**：所有算子必须通过 `cv_wrappers.算子名(img, **params)` 的形式调用。
* **纯净性**：函数内不要包含 `import` 语句，不要定义全局变量。

### Strategy & Best Practices / 策略建议
* **命名规范**：在 `trial.suggest` 中使用 `"{{算子名}}_{{参数名}}"` 的命名方式，防止参数冲突。
* **流程合理性**：遵循经典的 CV 顺序（例如：去噪 -> 增强 -> 边缘检测）。
* **边界保护**：确保每个参数都在 Schema 给定的 `range` 范围内，不要越界。

### Provided Schema / 算子库文档
{tool_schemas}

### Output Format / 输出格式要求
你必须直接返回代码块，不要包含冗长的解释。代码结构应如下：

```python
def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:
    # 1. 参数采样 (基于 Schema 范围)
    # 2. 图像处理流程
    # 3. 返回处理结果
    ...
```

## 示例 (Few-Shot)

**User Input:** "我想给一张老照片去噪，但要保留建筑的轮廓，不要变模糊。"

**Assistant Response:**
```python
def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:
    # 针对“保边降噪”的需求，选择双边滤波 (Bilateral_Filter)
    
    # 1. 采样参数
    d = trial.suggest_int("Bilateral_Filter_d", 1, 9)
    s_color = trial.suggest_float("Bilateral_Filter_sigma_color", 10.0, 150.0)
    s_space = trial.suggest_float("Bilateral_Filter_sigma_space", 10.0, 150.0)
    
    # 2. 执行处理
    out = cv_wrappers.Bilateral_Filter(
        img, 
        d=d, 
        sigma_color=s_color, 
        sigma_space=s_space
    )
    
    return out
```
        """
        return prompt.strip()

    def _extract_code_block(self, llm_response: str) -> str:
        """
        从LLM回复中提取纯Python代码块（剥离Markdown格式和多余解释）
        
        :param llm_response: LLM原始回复文本
        :return: 纯Python代码字符串（仅process函数）
        :raises RuntimeError: 无法从回复中提取合法代码块时抛出异常
        """
        # 正则匹配```python ... ```代码块
        code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
        match = code_pattern.search(llm_response)
        
        if not match:
            # 兜底匹配：无markdown格式时直接提取process函数
            func_pattern = re.compile(r'def process\(img.*?, *trial.*?\):\s*(.*?)(?=\ndef|\n$)', re.DOTALL)
            match = func_pattern.search(llm_response)
            if not match:
                logger.error(f"无法从LLM回复中提取代码块，原始回复：{llm_response}")
                raise RuntimeError("LLM回复未包含合法的process函数代码块")
        
        code = match.group(1).strip()
        logger.info("成功从LLM回复中提取process函数代码")
        return code
    
    def generate_prompt(self, 
        user_intent: str = '', 
        init_details: str = '', 
        previous_errors: str = None,
    ) -> str:
        user_prompt = ''
        if user_intent:
            user_prompt += f"用户图像增强需求：{user_intent}\n"
        if init_details:
            user_prompt += f"原始图像量化信息：{init_details}\n"
        if previous_errors:
            user_prompt += f"上一轮代码执行错误信息：{previous_errors}\n请优先修复该错误再生成代码\n"
        logger.info(f"注入提示词：\n{user_prompt}")
        return user_prompt

    def generate_code_stream(self, 
        user_intent: str = '', 
        init_details: str = '', 
        previous_errors: str = None,
    ) -> Generator[tuple[str, str], None, None]:
        """
        核心方法：根据用户意图生成/修复带Optuna trial的图像增强代码
        
        :param user_intent: 用户的图像增强需求（如"提升低光照图像的对比度和清晰度"）
        :param plan_steps: 规划步骤（预留参数，目前暂未使用）
        :param previous_errors: 上一轮代码执行的错误信息（用于修复代码）
        
        :return STREAM.REASONING: 流式返回思考内容
        :return STREAM.CONENT: 流式返回正文内容
        :return FINISH: 返回清理后的代码
        """
        # 调用LLM生成代码（继承BaseAgent的带重试机制的LLM调用）
        logger.info("开始调用LLM生成图像增强代码")
        llm_response = ''
        for t, chunk in self._call_llm_stream(self.generate_prompt(
            user_intent, init_details, previous_errors
        )):
            yield f'STREAM.{t}', chunk
            if t == "CONTENT":
                llm_response += chunk
        
        # 提取并清洗代码块
        code = self._extract_code_block(llm_response)
        logger.info(f"成功生成代码：\n{code}")
        yield 'FINISH', code

    def generate_code(self, 
        user_intent: str = '', 
        plan_steps: list = [], 
        previous_errors: str = None,
    ) -> str:
        """
        核心方法：根据用户意图生成/修复带Optuna trial的图像增强代码
        
        :param user_intent: 用户的图像增强需求（如"提升低光照图像的对比度和清晰度"）
        :param plan_steps: 规划步骤（预留参数，目前暂未使用）
        :param previous_errors: 上一轮代码执行的错误信息（用于修复代码）
        :return: 可执行的Python代码字符串（含process函数）
        """
        
        # 调用LLM生成代码（继承BaseAgent的带重试机制的LLM调用）
        logger.info("开始调用LLM生成图像增强代码")
        llm_response = self._call_llm(self.generate_prompt(
            user_intent, plan_steps, previous_errors
        ))
        
        # 提取并清洗代码块
        code = self._extract_code_block(llm_response)
        logger.info(f"成功生成代码：\n{code}")
        return code

    def execute(self, user_intent: str = '', init_details: str = '', previous_errors: str = None) -> str:
        """
        实现父类BaseAgent的抽象execute方法，作为Agent对外的统一执行入口
        
        :param user_intent: 用户的图像增强意图
        :param init_details: 初始化基础量化信息
        :param previous_errors: 历史执行错误信息（用于代码修复）
        :return: 最终生成的可执行Python代码字符串
        """
        try:
            return self.generate_code(user_intent, init_details, previous_errors)
        except Exception as e:
            logger.error(f"CoderAgent执行失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"编码Agent生成代码失败：{str(e)}") from e
        
    def execute_stream(
        self, 
        user_intent: str = '', 
        init_details: str = '', 
        previous_errors: str = None
    ) -> Generator[tuple[str, str], None, None]:
        """
        :return STREAM.REASONING: 流式返回思考内容
        :return STREAM.CONENT: 流式返回正文内容
        :return FINISH: 返回清理后的代码
        """
        try:
            for t, chunk in self.generate_code_stream(user_intent, init_details, previous_errors):
                yield t, chunk
        except Exception as e:
            logger.error(f"CoderAgent执行失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"编码Agent生成代码失败：{str(e)}") from e