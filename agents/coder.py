from agents.base_agent import BaseAgent
from tools import global_registry
from typing import Generator, Literal

import logging

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
        allow_learning: bool = True,
        operator_preference: Literal[
            "traditional_only",
            "prefer_traditional",
            "prefer_learning",
            "learning_only"
        ] = "prefer_traditional",
        additional_imports: list[str] | None = None,
        **kwargs
    ):
        """
        初始化编码Agent，继承BaseAgent的LLM通信能力
        
        :param llm_client: 大模型客户端实例（如openai.Client、通义千问客户端等）
        :param model_name: 使用的大模型名称，默认gpt-4o-mini（兼顾效率和代码生成能力）
        :param temperature: 生成温度，低温度保证代码逻辑稳定性（0.0-0.2为宜）
        """
        self.low_res = low_res
        self.allow_learning = bool(allow_learning)
        self.operator_preference = self._normalize_operator_preference(
            operator_preference, self.allow_learning
        )
        self.learning_enabled_for_code = (
            self.allow_learning and self.operator_preference != "traditional_only"
        )
        dynamic_imports = [
            str(imp).strip() for imp in (additional_imports or [])
            if str(imp).strip()
        ]
        if not self.learning_enabled_for_code:
            blocked_roots = {"torch", "torchvision", "transformers", "diffusers", "modelscope", "huggingface_hub"}
            dynamic_imports = [
                imp for imp in dynamic_imports
                if imp.split(".", maxsplit=1)[0].strip().lower() not in blocked_roots
            ]
        self.additional_imports = ', '.join(
            f"`{imp}`" for imp in dynamic_imports
        ) if dynamic_imports else ''
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, self._build_system_prompt(), temperature, reasoning_effort, **kwargs)
        
        logger.info("CoderAgent 初始化完成，已加载全局算子注册表")

    def rebuild_system_prompt(self):
        self.system_prompt = self._build_system_prompt()

    @staticmethod
    def _normalize_operator_preference(
        value: str,
        allow_learning: bool
    ) -> Literal["traditional_only", "prefer_traditional", "prefer_learning", "learning_only"]:
        mode = str(value or "").strip().lower()
        alias = {
            "traditional_only": "traditional_only",
            "only_traditional": "traditional_only",
            "traditional": "traditional_only",
            "仅传统": "traditional_only",
            "prefer_traditional": "prefer_traditional",
            "traditional_preferred": "prefer_traditional",
            "偏好传统": "prefer_traditional",
            "prefer_learning": "prefer_learning",
            "prefer_deep_learning": "prefer_learning",
            "偏好深度学习": "prefer_learning",
            "learning_only": "learning_only",
            "only_learning": "learning_only",
            "deep_learning_only": "learning_only",
            "only_deep_learning": "learning_only",
            "仅深度学习": "learning_only",
        }
        normalized = alias.get(mode, "prefer_traditional")
        if not allow_learning:
            return "traditional_only"
        return normalized

    def _schema_filter_mode(self) -> Literal["all", "traditional_only", "learning_only"]:
        if self.operator_preference == "traditional_only":
            return "traditional_only"
        if self.operator_preference == "learning_only":
            return "learning_only"
        return "all"

    def _build_system_prompt(self) -> str:
        """
        构建专属系统提示词，明确代码生成的硬性规则和格式要求
        核心原则：让LLM生成可直接被Optuna调用、容错性强的process函数
        """

        schema_filter_mode = self._schema_filter_mode()
        schema_list = global_registry.get_schemas_for_llm(
            allow_learning=self.allow_learning,
            learning_mode=schema_filter_mode
        )
        schema_dump = global_registry.get_all_schemas_for_llm(
            allow_learning=self.allow_learning,
            learning_mode=schema_filter_mode
        )

        preference_guidance_map = {
            "traditional_only": "仅传统：你只能使用传统算子，禁止使用深度学习算子。",
            "prefer_traditional": "偏好传统：优先尝试传统算子；当传统方案明显不足时，再使用深度学习算子。",
            "prefer_learning": "偏好深度学习：优先尝试深度学习算子；若传统算子更稳定或更低成本，也可混合/回退。",
            "learning_only": "仅深度学习：你只能使用深度学习算子，禁止使用传统算子。"
        }
        preference_guidance = preference_guidance_map.get(
            self.operator_preference, preference_guidance_map["prefer_traditional"]
        )

        if self.learning_enabled_for_code:
            learning_libraries_text = (
                "`torch`, `torchvision`, `transformers`, `diffusers`, `modelscope`, "
                f"{self.additional_imports if self.additional_imports else '（无额外模块）'}"
            )
            learning_switch_status = "启用"
            learning_switch_policy = (
                "允许使用深度学习算子，但必须遵守本地模型优先与显存回落规则。"
                if self.operator_preference != "learning_only"
                else "当前为仅深度学习模式，禁止使用传统算子。必须基于深度学习算子完成流程，并遵守本地模型优先与显存回落规则。"
            )
        else:
            learning_libraries_text = (
                "当前会话已禁用深度学习处理，严禁使用 `torch` / `torchvision` / "
                "`transformers` / `diffusers` / `modelscope` 及任何需要下载/加载深度模型的模块。"
            )
            learning_switch_status = "禁用"
            learning_switch_policy = (
                "禁止生成任何需要深度学习推理/模型权重加载的处理流程。若需求必须依赖深度学习，请输出 NEED_NEW_TOOL 并在 description 明确“当前深度学习已禁用，无法执行”。"
            )

        schema_filter_notice_map = {
            "traditional_only": "Schema 已按“仅传统”过滤，仅包含传统算子。",
            "learning_only": "Schema 已按“仅深度学习”过滤，仅包含深度学习算子。",
            "all": "Schema 未过滤，包含传统+深度学习算子。"
        }
        schema_filter_notice = schema_filter_notice_map.get(schema_filter_mode, schema_filter_notice_map["all"])
        schema_empty_guard = ""
        if schema_filter_mode in ("traditional_only", "learning_only") and not schema_list:
            desired = "传统算子" if schema_filter_mode == "traditional_only" else "深度学习算子"
            schema_empty_guard = (
                f"当前偏好过滤后没有任何可用{desired}。你必须直接输出 `NEED_NEW_TOOL` JSON，"
                "并在 description 里明确“当前偏好过滤后可用算子为空”。"
            )
        elif schema_filter_mode in ("traditional_only", "learning_only"):
            schema_empty_guard = "仅档位过滤后仍有可用算子，必须严格在该集合内选型。"

        # 从全局注册表中提取所有CV算子的Schema（供LLM参考可用函数）
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
* **函数签名**：必须严格为 `def process(img: np.ndarray, trial: optuna.Trial, cache: dict) -> np.ndarray:`。
* **参数决策**：仔细研判用户需求，决定是否要对算子的特定参数进行调优。
    - 情况 A (需要调优)：必须使用 `trial` 对象获取参数
        - 例如：`d = trial.suggest_int("Bilateral_Filter_d", 1, 9)`
        - 例如：`sigma_color = trial.suggest_float("Bilateral_Filter_sigma_color", 10.0, 40.0)`
        - 例如：`ksize_median = trial.suggest_categorical("Median_Denoise_ksize", [3, 5, 7])`
    - 情况 B (不需要)：必须使用 **常数** 设置参数
        - 例如：`adjust_sigmoid_cutoff = 0.5`
    - 情况 C (多工具可选 & 无可调参数)：如果工具库有多个可以实现目标的算子，并且算子 **均没有** 可调优参数，应当使用 `trial` 暴露一个可用算子列表给 Optuna
        - 例如：`low_light_choice = trial.suggest_categorical("Low_Light_Tool_Choice", ["Zero_DCE_Ext_Enhance", "SCI_Low_Light_Enhance"])`
    - **禁止调优的参数类型**：`cache`、`device`、`model_dir`、字符串路径、模型ID等运行时/环境参数，必须使用常量或直接透传，不得 `trial.suggest_*`
    - 如果算子暴露了 `model_dir` 参数，它是运行时固定参数，会由系统根据算子 schema 中的 `source` / `repo_id` 元数据自动注入；除非用户明确提供自定义本地路径，否则不要在调用算子时手写 `model_dir`。
* **库访问**：你只能使用下列库：
    - 基础处理: `np` (numpy), `cv2` (opencv-contrib-python), `optuna`, `skimage` (scikit-image), `PIL` (pillow) 以及提供的算子库 `cv_wrappers`
    - 深度学习:
      {learning_libraries_text}
* **算子调用**：所有算子必须通过 `cv_wrappers.算子名(img, **params)` 的形式调用
* **纯净性**：函数内不要导入模块，不要使用 `import`, `__import__` 语句，不要定义全局变量
* **辅助函数**：允许编写辅助函数简化过程、提高可读性，辅助函数必须嵌套在process函数中
* **单例模式**: 代码会被多次执行，只需要加载一次的内容必须放置在 `cache` 字典中
* **本地模型优先**: 若调用深度学习算子且其参数包含 `model_dir`，运行时会按来源自动注入本地目录；不要在 `process` 中拼接 Hugging Face / ModelScope 缓存路径，也不要构造联网下载逻辑
* **深度学习开关**: 当前会话{learning_switch_status}深度学习处理。{learning_switch_policy}
* **算子偏好**: {preference_guidance}
* **运行时信息读取**: 可从 `cache.get("__runtime__", {{}})` 读取运行时偏好：
    - `preferred_device`: 用户选择的处理设备（如 `cuda` / `mps` / `cpu`）
    - `performance_profile`: 用户选择的性能档位（`fast` / `balanced` / `low_memory`）
    - `device_info`: 系统设备信息摘要
* **显存回落兼容**: 若调用高显存占用算子（如含 `tile_size` / `patch_size` / `batch_size` 参数），必须优先把这些参数显式暴露并传给算子，避免写死在函数内部，便于运行时自动回落机制接管。

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
{schema_filter_notice}
{schema_empty_guard}
{schema_dump}

### Output Format / 输出格式要求
#### 格式 A：输出处理代码 (情况 A)
你必须直接返回代码块，不要包含冗长的解释。代码结构应如下：

```python
def process(img: np.ndarray, trial: optuna.Trial, cache: dict) -> np.ndarray:
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
def process(img: np.ndarray, trial: optuna.Trial, cache: dict) -> np.ndarray:
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
        
        return self._extract_code(llm_response, "process")
    
    def generate_prompt(self, 
        user_intent: str = '', 
        init_details: str = '', 
        previous_errors: str = None,
    ) -> str:
        user_prompt = ''
        if user_intent:
            user_prompt += f"{user_intent}"
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
        try:
            code = self._extract_code_block(llm_response)
            logger.info(f"成功生成代码：\n{code}")
            yield 'FINISH', code
        except:
            try:
                res_json = self._extract_json(llm_response)
                logger.info(f"要求生成工具：\n{res_json}")
                yield 'FINISH', res_json
            except:
                raise RuntimeError("LLM 回复既不包含合法的 JSON 工具请求，也未包含合法的 process 函数代码块")

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
        try:
            code = self._extract_code_block(llm_response)
            logger.info(f"成功生成代码：\n{code}")
            return code
        except:
            try:
                res_json = self._extract_json(llm_response)
                logger.info(f"要求生成工具：\n{res_json}")
                return res_json
            except:
                raise RuntimeError("LLM 回复既不包含合法的 JSON 工具请求，也未包含合法的 process 函数代码块")

    def execute(self, user_intent: str = '', init_details: str = '', previous_errors: str = None) -> str | dict:
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
    ) -> Generator[tuple[str, str | dict], None, None]:
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
