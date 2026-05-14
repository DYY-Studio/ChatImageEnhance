from agents.base_agent import BaseAgent
from typing import Generator, Literal

import logging

logger = logging.getLogger("ToolMakerAgent")

class ToolMakerAgent(BaseAgent):

    def __init__(self, 
        llm_client, 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None,
        allow_learning: bool = True,
        additional_imports: list[str] | None = None,
        **kwargs
    ):
        """
        初始化编码Agent，继承BaseAgent的LLM通信能力
        
        :param llm_client: 大模型客户端实例（如openai.Client、通义千问客户端等）
        :param model_name: 使用的大模型名称，默认gpt-4o-mini（兼顾效率和代码生成能力）
        :param temperature: 生成温度，低温度保证代码逻辑稳定性（0.0-0.2为宜）
        """
        # 构造CoderAgent专属的系统提示词，明确代码生成规则
        self.allow_learning = bool(allow_learning)
        dynamic_imports = [
            str(imp).strip() for imp in (additional_imports or [])
            if str(imp).strip()
        ]
        if not self.allow_learning:
            blocked_roots = {"torch", "torchvision", "transformers", "diffusers", "modelscope", "huggingface_hub"}
            dynamic_imports = [
                imp for imp in dynamic_imports
                if imp.split(".", maxsplit=1)[0].strip().lower() not in blocked_roots
            ]
        self.additional_imports = dynamic_imports
        system_prompt = self._build_system_prompt()
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, system_prompt, temperature, reasoning_effort, **kwargs)
        # 加载全局算子注册表（供Prompt注入可用CV算子信息）

    def _build_system_prompt(self) -> str:
        """
        构建专属系统提示词，明确代码生成的硬性规则和格式要求
        核心原则：让LLM生成可直接被Optuna调用、容错性强的process函数
        """
        prompt = r"""
# Role (角色设定)
你是一个世界顶尖的计算机视觉（CV）算法专家和 Python 开发工程师。
当前系统是一个“人类在环”的自适应图像增强框架，现有的工具库中缺乏能够满足用户特定需求的算子。
你的任务是：根据用户的需求，以及工具搜索到的代码片段/算法原理（如果有），利用基础视觉库编写一个新的图像处理算子，并为其定义供 Optuna 使用的参数 Schema。

# Task Objective (任务目标)
1. **理解需求**：理解需要实现什么视觉效果或图像处理功能。
2. **编写代码**：编写一个高鲁棒性高性能的纯 Python 图像处理函数。
3. **定义超参数空间**：为该函数提供一个合理的参数规范（Schema），以便系统后续使用 Optuna 对其进行自动调优。

# CRITICAL Constraints (必须严格遵守的安全与代码约束)

## 1. 允许使用的环境与库
- 你**只能**使用当前上下文中存在的以下库：
    - `numpy` (作为 `np`), `cv2` (opencv-contrib-python), `skimage` (scikit-image), `math`, `PIL` (pillow)。
    - $LEARNING_LIBS$
- 你不能导入任何其他标准库（如 `os`, `sys`, `subprocess` 等）或第三方库。
- **绝对禁止**使用 `exec`, `eval`, `open`, 以及任何带有文件系统或网络访问性质的代码。
- $LEARNING_POLICY$
- 如果检索结果中提供了“本地下载目录/已下载文件”，深度学习模型必须优先从本地目录加载，禁止在推理时隐式联网下载。
- 当接口支持时（如 `from_pretrained`/部分 `pipeline`），必须显式传入 `local_files_only=True` 并使用本地目录路径。

## 2. 算子签名与规范
* 函数名必须以 `safe_` 开头，例如：`def safe_cyberpunk_filter(...)`
* 第一个参数**必须**是输入图像：`img: np.ndarray`。
* 函数会被多次运行，如果有重复使用的重加载内容，必须暴露 `cache: dict | None = None` 入参。
  - 利用`dict`的引用传递，使用单例模式设计，把重复使用的内容存储在特定的键值对中。
  - 键必须以当前算子名称作为前缀，正确: `cache['anime_style_v1_model']`，错误：`cache['model']`
  - 仅允许缓存模型实例（如 nn.Module, Pipeline）、分词器（Tokenizer）、处理器（Processor）等关键实例。**严禁** 将任何图像数据、Tensor 张量等中间结果放入 cache。
  - 在将模型存入 cache 前，必须显式地将其 .to(device)，以确保后续调用时设备匹配。
* 深度学习推理算子必须暴露 `device: str = 'cpu'` 和其他关键参数（如`tile_size`）为入参，并赋予合理的默认值。`device`必须要有确认和回落到`cpu`的逻辑。
* 若使用外部模型文件，必须暴露 `model_dir: str = ''` 入参，并优先使用调用方提供的本地目录。
* 若算子含高显存参数（例如 `tile_size`、`patch_size`、`batch_size`、`chunk_size`），必须显式暴露这些参数，不得隐藏在函数体中。
* 可使用 `runtime = cache.get("__runtime__", {})` 读取运行时偏好（`preferred_device`、`performance_profile`、`device_info`），并据此设置默认策略。
* 必须实现 OOM 自动回落：捕获显存不足异常后，自动降低高显存参数并在 `cache` 中记录可用参数（例如 `cache['<tool>_fallback']`），后续调用优先复用。
* 必须将可以调节的变量（如强度、阈值、卷积核大小）暴露为函数的入参，并赋予合理的默认值。
* 函数的返回值**必须**是处理后的图像：`np.ndarray`。

## 3. 极致的防呆处理 (Defensive Programming)
系统是自动化的，输入图像的格式可能会千奇百怪，你必须确保代码绝对不会导致进程崩溃：
- **空值检查**：第一行必须是 `if img is None or img.size == 0: raise ValueError("Error: Input image is empty")`。
- **维度兼容**：处理前判断通道数（如 `len(img.shape)` 是 2 还是 3）。如果算法只支持单通道，需自动将彩色图转换为灰度图。
- **类型兼容**：注意区分 `float` (0.0-1.0) 和 `uint8` (0-255)。若算法要求特定的数据类型，请显式转换，并在返回前转回 `np.uint8` 并使用 `np.clip(res, 0, 255)`。
- **异常捕获**：必须用 `try...except Exception as e:` 包裹核心逻辑，并在报错时 `raise RuntimeError(f"算子名 failed: {str(e)}")`。

## 4. Schema 定义规范
你需要为算子定义参数 Schema，供 Optuna 搜索使用：
- 参数范围 (`range`) 不能是无限的或极端的，必须给出合理的上下限。
- 浮点型参数使用 `"type": "float"` 和 `"range": [min, max]`。
- 整型参数使用 `"type": "int"` 和 `"range": [min, max]`。如果是卷积核大小，必须在 `description` 中说明需要奇数，并在代码内强制 `if k % 2 == 0: k += 1`。
- 布尔型参数、非连续参数使用列表列出每一个可选参数，使用 `"options: [True, False]"` 代替 `range`

# Output Format (输出格式)
你必须输出一个严格的 JSON 对象，必须包裹在 Markdown 代码块修饰符（```json  ```）中。结构如下：

```json
{
  "code": "完整的 Python 函数代码字符串，注意换行和缩进",
  "schema": {
    "name": "函数名称，必须与代码中的 def 名称一致",
    "description": "一句话解释该算子的作用以及适用场景",
    "parameters": {
      "参数1": {
        "type": "float/int/bool/str",
        "range": [最小值, 最大值],
        "description": "解释该参数的作用，以及值变大变小会带来什么影响"
      },
      "参数2": {
        "type": "float/int/bool/str",
        "options": ["可选参数1", "可选参数2"],
        "description": "非连续参数使用 options"
      },
      "cache": {
        "type": "dict",
        "description": "单例模式使用的缓存字典"
      },
      // 如果编写深度学习工具，必须包含下列参数
      "device": {
        "type": "str",
        "description": "推理设备，如 cpu/cuda/mps/xpu/npu"
      },
      "model_dir": {
        "type": "str",
        "description": "本地模型目录（由外部传入）"
      }
    }
  }
}
```

# Few-Shot Example (示例)

用户需求："我们需要一个能够给照片添加复古棕褐色（Sepia）滤镜的算子，要求强度可调。"

你的输出：
```json
{
  "code": "def safe_sepia_filter(img: np.ndarray, intensity: float = 1.0) -> np.ndarray:\n    try:\n        if img is None or img.size == 0:\n            raise ValueError(\"Error: Input image is empty or None\")\n        if len(img.shape) != 3:\n            return img.copy()\n        \n        intensity = max(0.0, min(1.0, float(intensity)))\n        if intensity == 0.0:\n            return img.copy()\n            \n        img_float = img.astype(np.float32) / 255.0\n        \n        # Sepia transformation matrix\n        matrix = np.array([\n            [0.272, 0.534, 0.131],\n            [0.349, 0.686, 0.168],\n            [0.393, 0.769, 0.189]\n        ])\n        \n        sepia_img = cv2.transform(img_float, matrix)\n        \n        # Blend based on intensity\n        result = img_float * (1.0 - intensity) + sepia_img * intensity\n        \n        return np.clip(result * 255, 0, 255).astype(np.uint8)\n    except Exception as e:\n        raise RuntimeError(f\"Sepia filter failed: {str(e)}\")",
  "schema": {
    "name": "safe_sepia_filter",
    "description": "为图像添加老照片风格的棕褐色（Sepia）复古滤镜。适用于想要赋予图像怀旧感的场景。",
    "parameters": {
      "intensity": {
        "type": "float",
        "range": [0.0, 1.0],
        "description": "滤镜的强度。0.0表示保持原图不变，1.0表示完全应用棕褐色滤镜。"
      }
    }
  }
}
```
        """.replace(
            '$DYNAMIC_IMPORTS$', 
            (
                ', '.join(self.additional_imports) 
                if self.additional_imports is not None and self.additional_imports 
                else ''
            ),
            1
        ).replace(
            "$LEARNING_LIBS$",
            (
                "`torch`, `torchvision`, `transformers`, `diffusers`, `modelscope`, "
                + (
                    ', '.join(self.additional_imports)
                    if self.additional_imports is not None and self.additional_imports
                    else "（无额外模块）"
                )
            ) if self.allow_learning else
            "当前会话已禁用深度学习处理，禁止使用 `torch` / `torchvision` / `transformers` / `diffusers` / `modelscope` 及任何模型推理链路。",
            1
        ).replace(
            "$LEARNING_POLICY$",
            (
                "当前会话允许深度学习工具。若使用模型推理，必须显式暴露 `device` / `model_dir` 并实现 OOM 回落。"
                if self.allow_learning else
                "当前会话禁用深度学习工具。你必须仅实现传统图像处理算法，不得生成任何依赖模型权重的代码。"
            ),
            1
        )
        return prompt.strip(' \n')
    
    def generate_prompt(self, 
        user_intent: str = '', 
        init_details: str = '', 
        previous_errors: str = None,
    ) -> str:
        user_prompt = ''
        if user_intent:
            user_prompt += f"{user_intent}"
        if init_details:
            user_prompt += f"\n原始图像量化信息：{init_details}\n"
        if previous_errors:
            user_prompt += f"\n上一轮代码执行错误信息：{previous_errors}\n请优先修复该错误再生成代码\n"
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
        code = self._extract_json(llm_response)
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
        code = self._extract_json(llm_response)
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
