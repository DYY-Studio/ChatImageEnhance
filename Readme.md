# AutoImageEnhance
一个实验性多Agents系统

由自然语言描述生成自动调优的CV图像处理管线

## 特色
* 高度兼容的调用机制，兼容任何OpenAI兼容API端点
  * 主要内容均通过基础角色 (`system`, `user`) 传入
  * 不需要工具调用支持
  * 可选的多模态支持
  * 没有连续对话
* 本地部署模型优化
  * 仅 `16K` 上下文即可运行主要功能
  * 如果要进行代码检索，需要提高至 `32K`
* 高度保守的代码运行机制
  * 严格的 `ast` 校验
  * `globals` 受限的 `exec` 注入
  * 预封装的安全算子
  * 没有命令行执行
* 无费用的联网搜索
  * 通过 GitHub REST API 实现代码检索
  * 通过 HuggingFace API, ModelScope API 实现深度学习模型查询
  * 通过 DuckDuckGo / Bing 实现简易网页检索
* 可扩展性
  * 支持添加自定义算子
  * 支持LLM自动算子编写

## 主要功能
### 图像分析
由 LLM 根据客观量化数据（以及可选的图像输入）对图像存在的问题进行诊断

给出一个可选的修改方向
### 自动管线编写与调优
根据用户的要求自动编写图像处理管线、奖励函数，并进行调优

### 联网搜索与自动算子编写
缺少工具时，可以通过 GitHub REST API 进行联网搜索，然后自行编写安全算子

如果用户满意，可以将该算子保存到本地以便之后使用

## 运行方法
```shell
streamlit run app.py
```

## TODO
* 使用RAG保存算子描述，简化注入的算子数
* 优化深度学习模型的生命周期控制，解决内存泄漏问题
* 提高在受限环境搜索深度学习模型的成功率
* 优化LLM选择搜索关键词的准确度

## 架构描述

### 五大Agent
#### PlannerAgent
* （视觉模型）负责根据图像内容和客观量化参数分析图像存在的问题，并给用户给出推荐的优化方向和建议提示词

#### EvaluatorAgent
* 负责将用户的自然语言输入转换为评价代码用于本轮本地Optuna调优
  ```python
  def evaluate(img: np.ndarray) -> float:
    pass
  ```

#### CoderAgent
* 输入：自然语言输入，评价代码
* 根据用户的自然语言输入，参考评价代码，编写可运行的图像处理管线函数
  ```python
  def process(img: np.ndarray, trial: optuna.Trial) -> np.ndarray:
    pass
  ```

#### SearcherAgent
* 输入：缺少的工具的描述
* 根据缺少的工具的描述，利用 GitHub REST API 搜索相关代码

#### ToolMakerAgent
* 输入：缺少的工具的描述，相关代码
* 根据输入信息编写健壮的图像处理工具

## 运行方式
1. 用户输入图像和自然语言描述（或使用PlannerAgent生成提示词）
2. EvaluatorAgent根据用户的自然语言输入编写评价函数，指导本轮Optuna调优
3. CoderAgent根据用户的自然语言输入编写主处理管线。
   1. CoderAgent发现本地工具无法完成任务，发起工具编写请求
   2. SearcherAgent根据要求搜索GitHub，寻找相关代码
   3. ToolMakerAgent根据要求和相关代码，编写健壮的工具
4. 用户检查结果是否符合预期，输入下一步操作的自然语言描述
5. 回到第一步

## 架构图
```plaintext
AutoImageEnhance/
├── app.py                      # 负责前端交互 (Streamlit UI) 
│
├── main_cli.py                 # 纯命令行入口 (占位，目前没有计划)
│
├── core/                       # 核心业务逻辑层
│   ├── __init__.py
│   ├── orchestrator.py         # 主控调度器：管理 LLM 与 Optuna 的双循环机制
│   ├── evaluator.py            # 图像质量评估器：定义基本图像质量评估指标
│   ├── searcher.py             # 代码检索器：封装代码检索流程
│   ├── model_assets.py         # 模型过滤器常量
│   └── optimizer.py            # 贝叶斯优化器：封装 Optuna 逻辑
│
├── agents/                     # 大模型 Agent 层
│   ├── __init__.py
│   ├── base_agent.py           # 基础 Agent 类 (封装 LLM API 调用)
│   ├── planner.py              # 规划者：分析图像，给出图像可改进的内容，并给用户推荐提示词
│   ├── evaluator.py            # 评估者：将提示词转化为适用于 Optuna Study 的奖励函数
│   ├── seacher.py              # 检索者：利用 GitHub REST API / HF API / ModelScope API 或 DuckDuckGo / Bing 检索相关的代码实现和模型
│   ├── toolmaker.py            # 工具编写者：按编码者的需求和检索者收集到的信息编写工具
│   └── coder.py                # 编码者：将提示词转化为带有 Optuna trial 的 Python 代码
│
├── tools/                      # 传统计算机视觉库
│   ├── __init__.py             # 全局算子注册
│   ├── cv_wrappers.py          # 经过防呆处理的 OpenCV 封装函数库
│   ├── skimage_wrappers.py     # 经过防呆处理的 scikit-image 封装函数库
│   ├── learning_wrappers.py    # 经过防呆处理的深度学习封装库
│   └── registry.py             # 算子注册表：动态将 cv_wrappers 导出为 LLM 可读的 JSON Schema
│
├── sandbox/                    # 运行时与沙盒环境
│   ├── __init__.py
│   ├── code_checker.py         # AST代码安全检查
│   ├── runtime_dependencies.py # 动态依赖解析安装器
│   ├── safe_os_path.py         # 安全os.path封装
│   └── executor.py             # 动态代码安全执行器 (负责 exec 运行 LLM 生成的代码)
│
├── memory/                     # 记忆与经验库
│   ├── __init__.py
│   └── experience_db.py        # 封装 ChromaDB
│
├── models/                     # 内置深度学习模型
│   ├── AestheticScorePredictor.py # CLIP+MLP 美学分数预测
│   ├── FFDNet.py
│   ├── ImageAdaptive3DLUT.py
│   ├── SCI.py
│   ├── SepLUT.py
│   └── ZeroDCE.py              # ZeroDCE++，修改为输出LUT，由CPU套用
│
├── utils.py                    # 通用工具            
│
└── requirements.txt            # 项目依赖
```

## Git 写作指南
[Git 写作指南](git_guide.md)