# AutoImageEnhance
# 运行方法
`streamlit run app.py`
# 架构描述
### 宏观微观双循环

让大模型（LLM）负责它最擅长的**逻辑与结构（离散空间）**，让传统数学工具（Optuna）负责它最擅长的**数值调参（连续空间）**。

整个系统只需要**一个大模型 Agent** 和**一个本地寻优引擎**，彻底抛弃复杂的 Agent 间对话。

#### 1. 外循环 (Outer Loop) —— LLM 拓扑突变 (类似 autoresearch)
* **动作**：LLM 编写完整的 Python 执行函数代码（process函数）
* **频率**：低频。每当内循环遇到瓶颈时，LLM 才会被唤醒一次。
* **原理**：利用 LLM 的先验知识（例如“边缘检测前最好先做高斯模糊”）和 `autoresearch` 的进化思想，随机或有逻辑地“变异”这个管线函数（比如：增加一个锐化节点，或者把均值滤波换成双边滤波）。

#### 2. 内循环 (Inner Loop) —— 本地贝叶斯优化 (Optuna)
* **动作**：一旦 LLM 确定了执行函数，**断开与大模型的连接**。本地的 Optuna 接管系统。
* **频率**：高频。在本地 CPU/GPU 上以毫秒级速度疯狂运行。
* **原理**：Optuna 在这个固定的管线下，自动调节 `kernel_size`、`clipLimit` 等参数，快速迭代 30~50 次，找到这套管线下的**绝对最高分**。

#### 3. 评估器 (The Evaluator) —— 唯一的判官
* 这是一个纯 Python 编写的客观打分函数（清晰度、对比度、噪声评估）。内循环靠它指引参数收敛，外循环靠它决定这套管线是“保留 (Keep)”还是“丢弃 (Discard)”。
* 保留了在此处追加多模态模型进行主观评判的可能性

# 架构图
```plaintext
AutoImageEnhance/
├── app.py                      # 负责前端交互 (Streamlit UI或其他) 
├── main_cli.py                 # 纯命令行入口 (方便后期自动化测试)
├── core/                       # 核心业务逻辑层
│   ├── __init__.py
│   ├── orchestrator.py         # 主控调度器：管理 LLM 与 Optuna 的双循环机制
│   ├── evaluator.py            # 图像质量评估器：定义奖励函数
│   └── optimizer.py            # 贝叶斯优化器：封装 Optuna 逻辑
├── agents/                     # 大模型 Agent 层
│   ├── __init__.py
│   ├── base_agent.py           # 基础 Agent 类 (封装 LLM API 调用)
│   ├── planner.py              # 规划者：理解需求，选择合适算子（占位，目前不使用）
│   └── coder.py                # 编码者：将规划转化为带有 Optuna trial 的 Python 代码
├── tools/                      # 传统计算机视觉库
│   ├── __init__.py             # 全局算子注册
│   ├── cv_wrappers.py          # 经过防呆处理的 OpenCV 封装函数库
│   └── registry.py             # 算子注册表：动态将 cv_wrappers 导出为 LLM 可读的 JSON Schema
├── sandbox/                    # 运行时与沙盒环境
│   ├── __init__.py
│   ├── code_checker.py         # AST代码安全检查
│   └── executor.py             # 动态代码安全执行器 (负责 exec 运行 LLM 生成的代码)
├── memory/                     # 记忆与经验库
│   ├── __init__.py
│   └── experience_db.py        # 封装 ChromaDB，存储成功案例
└── requirements.txt            # 项目依赖
```

# Git 协作指南

为了保证代码互不干扰、高效协作，我们采用 Git 进行版本控制。如果你是第一次使用 Git，请按照本指南操作。

使用 Git 可有效避免协作中出现的各种冲突问题，并且有足够的余地进行回退等操作。

## 1. 准备工作

在开始之前，请确保你的电脑已安装 [Git](https://git-scm.com/)，并完成以下配置：

```bash
# 设置你的姓名和邮箱（请使用 GitHub/Gitee 账号的邮箱）
git config --global user.name "你的名字"
git config --global user.email "your_email@example.com"
```

## 2. 核心工作流：六步走

为了防止直接修改主代码导致混乱，我们遵循 **“分支开发，Pull Request 汇总”** 的原则。

下列步骤在**JetBrain**、**VS Code**等主流编辑器中均有可视化界面，请参照进行。

### 第零步：克隆项目
如果你本地还没有代码，请先从本仓库进行`clone`
```bash
git clone <URL>
```

### 第一步：获取最新代码
在开始任何工作前，先同步远程仓库的最新进展。
```bash
git checkout main
git pull origin main
```

### 第二步：创建功能分支
**不要直接在 main 分支修改代码！** 请为你的任务创建一个新分支。
```bash
# 分支名建议：feature-任务名 (例如: feature-login)
git checkout -b feature-task
```

### 第三步：编写代码与提交
当你完成了一个小阶段的工作，将更改保存到本地 Git 仓库。
```bash
# 1. 查看改动了哪些文件
git status

# 2. 将改动添加到暂存区
git add .

# 3. 提交改动，并写下清晰的说明
git commit -m "完成登录页面的 UI 设计"
```

### 第四步：推送分支到远程
将你的本地分支上传到云端仓库。
```bash
git push origin feature-yourname-task
```

### 第五步：发起 Pull Request (PR)
1. 打开网页版 GitHub/Gitee 仓库页面。
2. 点击 **"Compare & pull request"** 按钮。
3. 检查代码改动，确认无误后点击 **"Create pull request"**。
4. 在群里通知队友进行 Code Review（代码审核）。

### 第六步：合并与删除分支
一旦 PR 被通过并合并到 `main`，你就可以回到本地删掉这个分支，并准备开始下一个任务。
```bash
git checkout main
git pull origin main
git branch -d feature-yourname-task
```

---

## 3. 必知必会：黄金守则

1.  **小步快跑**：不要写了一周代码才提交一次。建议每完成一个完整的小功能（如：修复了一个 Bug、写好了一个函数）就进行一次 `commit`。
2.  **注释清晰**：`commit` 的消息要让队友一眼看出你改了什么。
    * ❌ 错误示范：`git commit -m "fix"`
    * ✅ 正确示范：`git commit -m "修复了导航栏在移动端显示不全的问题"`
3.  **先拉后推**：在 `push` 代码前，养成先 `pull` 的习惯，减少冲突概率。

---

## 4. 遇到“冲突(Conflict)”怎么办？

当你和队友同时修改了同一个文件的同一行时，Git 会提示 `Conflict`。别慌：
1. 打开提示冲突的文件，你会看到 `<<<<<<< HEAD` 这种标记。
2. **手动选择**保留哪部分代码，删除 Git 自动生成的标记。
3. 重新执行 `git add .` 和 `git commit`。
4. 如果搞砸了，随时求助其他组员，千万不要带冲突强制推送！

---

## 常用命令速查表

| 命令 | 作用 |
| :--- | :--- |
| `git clone <url>` | 克隆项目到本地 |
| `git pull origin main` | 拉取最新的主分支代码 |
| `git checkout -b <name>` | 创建并切换到新分支 |
| `git status` | 查看当前文件状态 |
| `git add .` | 添加所有改动到暂存区 |
| `git commit -m "msg"` | 提交改动并备注 |
| `git push origin <name>` | 推送分支到服务器 |
| `git log --oneline` | 查看简洁的提交历史 |

---