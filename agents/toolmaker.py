from agents.base_agent import BaseAgent
from typing import Generator, Iterable, Literal

import logging

logger = logging.getLogger("ToolMakerAgent")

class ToolMakerAgent(BaseAgent):
    _LEARNING_IMPORT_ROOTS = {
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "modelscope",
        "huggingface_hub",
    }

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
        self.additional_imports = self._normalize_additional_imports(additional_imports)
        system_prompt = self._build_system_prompt()
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, system_prompt, temperature, reasoning_effort, **kwargs)
        # 加载全局算子注册表（供Prompt注入可用CV算子信息）

    def _normalize_additional_imports(self, imports: Iterable[str] | None) -> list[str]:
        normalized: list[str] = []
        for imp in imports or []:
            token = str(imp).strip()
            if not token:
                continue
            root = token.split(".", maxsplit=1)[0].strip().lower()
            if (not self.allow_learning) and root in self._LEARNING_IMPORT_ROOTS:
                continue
            if token not in normalized:
                normalized.append(token)
        return normalized

    def set_additional_imports(self, imports: Iterable[str] | None):
        """Refresh the system prompt after search has resolved runtime deps."""
        self.additional_imports = self._normalize_additional_imports(imports)
        self.system_prompt = self._build_system_prompt()

    def _allowed_optional_libraries_text(self) -> str:
        extras = [f"`{imp}`" for imp in self.additional_imports]
        if self.allow_learning:
            libs = [
                "`torch`",
                "`torchvision`",
                "`transformers`",
                "`diffusers`",
                "`modelscope`",
            ]
            if extras:
                libs.extend(extras)
            return ", ".join(libs)

        if extras:
            return "深度学习处理已禁用；额外允许的非深度学习模块：" + ", ".join(extras)
        return "当前会话已禁用深度学习处理，且无额外模块。"

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
- 系统已经设置 `HF_HOME` 和 `MODELSCOPE_CACHE`，repo-id 驱动的加载方式应优先依赖这些缓存根目录。
- 对 Hugging Face / ModelScope 的标准接口（如 `from_pretrained(repo_id, local_files_only=True)`、`pipeline(..., model=repo_id)`、`Model.from_pretrained(repo_id)`），不要暴露 `model_dir`，应在函数内部使用固定 `repo_id` 常量并启用离线/本地优先参数。
- 只有当模型加载必须依赖本地文件夹或文件路径时，才暴露 `model_dir: str = ''`，例如：自定义 `torch.load(model_dir + '/xxx.pth')`、ONNX/权重文件、GitHub 资产、或必须以本地目录作为 `from_pretrained(model_dir)` 输入的仓库快照。
- 如果检索附加信息列出了“已下载的外部资产文件”，这些文件位于运行时注入的 `model_dir` 下；代码应按文件名相对 `model_dir` 读取，禁止假设权重已在当前工作目录。
- 禁止在推理时隐式联网下载；不要硬编码 Hugging Face / ModelScope 缓存路径。

## 2. 算子签名与规范
* 函数名必须以 `safe_` 开头，例如：`def safe_cyberpunk_filter(...)`
* 第一个参数**必须**是输入图像：`img: np.ndarray`。
* 当深度学习模型结构无法通过库函数直接加载时，允许在 `safe_` 函数前定义必要的顶层辅助类或辅助函数，例如 `torch.nn.Module` 子类；这些辅助类必须只服务于当前算子，不得执行文件系统扫描、网络访问或进程级操作。
  - 辅助类允许定义 `__init__` 和 `forward` 等普通模型方法，初始化父类时使用 `super().__init__()`。
  - 优先使用 `torch.nn.Module`、`torch.nn.Sequential`、`torch.nn.Conv2d` 等完整命名；如确需导入别名，只能从允许库中导入，例如 `import torch.nn as nn`、`import torch.nn.functional as F`。
  - 模型实例、预处理器和权重必须在 `safe_` 函数内部按 `cache` 单例加载，禁止在模块顶层实例化大模型或读取权重。
* 函数会被多次运行，如果有重复使用的重加载内容，必须暴露 `cache: dict | None = None` 入参。
  - 利用`dict`的引用传递，使用单例模式设计，把重复使用的内容存储在特定的键值对中。
  - 键必须以当前算子名称作为前缀，正确: `cache['anime_style_v1_model']`，错误：`cache['model']`
  - 仅允许缓存模型实例（如 nn.Module, Pipeline）、分词器（Tokenizer）、处理器（Processor）等关键实例。**严禁** 将任何图像数据、Tensor 张量等中间结果放入 cache。
  - 在将模型存入 cache 前，必须显式地将其 .to(device)，以确保后续调用时设备匹配。
* 深度学习推理算子必须暴露 `device: str = 'cpu'` 和其他关键参数（如`tile_size`）为入参，并赋予合理的默认值。`device`必须要有确认和回落到`cpu`的逻辑。
* 仅当加载逻辑必须使用本地文件夹/文件路径时，才暴露 `model_dir: str = ''` 入参，并优先使用调用方提供的本地目录；repo-id 驱动的 `from_pretrained` / `pipeline` 不需要暴露 `model_dir`。
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
  "code": "完整的 Python 代码字符串，可包含必要的辅助类/辅助函数，但必须包含一个 safe_ 算子函数，注意换行和缩进",
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
      // 如果编写深度学习工具，必须包含 device
      "device": {
        "type": "str",
        "description": "推理设备，如 cpu/cuda/mps/xpu/npu"
      },
      // 只有当工具必须直接读取本地模型文件/目录时，才包含 model_dir
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
            "$LEARNING_LIBS$",
            self._allowed_optional_libraries_text(),
            1
        ).replace(
            "$LEARNING_POLICY$",
            (
                "当前会话允许深度学习工具。若使用模型推理，必须显式暴露 `device` 并实现 OOM 回落；仅本地文件路径加载场景需要暴露 `model_dir`。"
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
