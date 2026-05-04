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