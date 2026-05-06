from agents.base_agent import BaseAgent
from typing import Generator, Literal

from github import Github
from github.ContentFile import ContentFile

import logging
import yaml

logger = logging.getLogger("SearcherAgent")

class SearcherAgent(BaseAgent):

    def __init__(self, 
        llm_client, 
        model_name: str = "gpt-4o-mini", 
        github_client: Github | None = None, 
        temperature: float = 0.1,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None,
        **kwargs
    ):
        """
        初始化编码Agent，继承BaseAgent的LLM通信能力
        
        :param llm_client: 大模型客户端实例（如openai.Client、通义千问客户端等）
        :param model_name: 使用的大模型名称，默认gpt-4o-mini（兼顾效率和代码生成能力）
        :param github_client: 初始化完成的PyGithub绑定，如果传入None，则本Agent不工作
        :param temperature: 生成温度，低温度保证代码逻辑稳定性（0.0-0.2为宜）
        """
        # 构造CoderAgent专属的系统提示词，明确代码生成规则
        system_prompt = self._build_system_prompt()
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, system_prompt, temperature, reasoning_effort, **kwargs)

        self.github_client = github_client
        github_client.per_page = 10
        self.curr_repo = None
        self.query_cache: dict[str, str] = {}

        logger.info("CoderAgent 初始化完成，已加载全局算子注册表")

    def _search_github_repos(self, query: str) -> str:
        if query not in self.query_cache:
            self.query_cache.clear()
            yaml_text = yaml.dump([
                {
                    'name': repo.full_name,
                    'desc': repo.description,
                    'stars': repo.stargazers_count,
                    'forks': repo.forks_count,
                    'lang': repo.language,
                    'updated': repo.updated_at.isoformat(),
                }
                for repo in self.github_client.search_repositories(query, 'stars', 'desc').get_page(0)
            ], allow_unicode=True, indent=2)
            self.query_cache[query] = yaml_text if yaml_text != '[]' else 'No result'
        return self.query_cache[query]

    def _get_repo_overview(self, repo_name: str) -> str:
        self.curr_repo = self.github_client.get_repo(repo_name)
        readme = self.curr_repo.get_readme()
        if readme: readme = readme.decoded_content.decode()
        else: readme = 'No readme found'
        root_files: list[ContentFile] = self.curr_repo.get_contents('')
        return yaml.dump({
            "readme": f"{readme[:4096]} ... (over 4096 chars)" if len(readme) > 4096 else readme,
            "root_files": [
                {
                    'path': file.path,
                    'size': file.size,
                    'type': file.type
                }
                for file in root_files
            ]
        }, allow_unicode=True, indent=2)

    def _list_directory(self, repo_name: str, path: str = "") -> str:
        if repo_name != self.curr_repo.full_name:
            self.curr_repo = self.github_client.get_repo(repo_name)
        files: list[ContentFile] = self.curr_repo.get_contents(path)
        if not isinstance(files, list):
            files = [files]
        return yaml.dump([
            {
                'path': file.path,
                'size': file.size,
                'type': file.type
            }
            for file in files
        ], allow_unicode=True, indent=2)

    def _read_file_content(self, repo_name: str, file_path: str, start_line: int = 0, end_line: int = -1) -> str:
        start_line = int(start_line)
        end_line = int(end_line)
        if repo_name != self.curr_repo.full_name:
            self.curr_repo = self.github_client.get_repo(repo_name)
        try:
            file_content =  self.curr_repo.get_contents(file_path).decoded_content.decode()
        except:
            return f'{file_path} text decode failed'
        file_lines = file_content.splitlines()

        if end_line < 0:
            if len(file_lines) >= start_line:
                return '\n'.join(file_lines[start_line:])
            else:
                return file_content
            
        if len(file_lines) < max(end_line - start_line, 0):
            return file_content
        elif len(file_lines) < max(end_line, 0):
            return '\n'.join(file_lines[:end_line - start_line])
        else:
            return '\n'.join(file_lines[start_line:end_line])
        
    def _submit_findings(self, code_snippets: str = '', dependencies: str = '', summary: str = ''):
        return {
            "code_snippets": code_snippets,
            "dependencies": dependencies,
            "summary": summary
        }

    def _build_system_prompt(self) -> str:
        """
        构建专属系统提示词，明确代码生成的硬性规则和格式要求
        核心原则：让LLM生成可直接被Optuna调用、容错性强的process函数
        """
        prompt = """
# Role: GitHub Code Searcher Agent
你是系统中负责“代码检索与评估”的资深研发工程师。
你的任务是利用 GitHub 搜索特定功能的实现代码，经过层层筛选、阅读和评估，最终提取出高质量的核心逻辑代码片段或算法步骤，并移交给 ToolMakerAgent。
代码片段或算法步骤需要能够在 Python 中进行等价实现。

# Objective
根据用户的自然语言需求（例如：“寻找一段可以将图片卡通化的代码”），在 GitHub 上定位最佳仓库，提取可运行的核心算法或函数。

# Available Tools
你必须严格按照逻辑顺序使用以下工具：
1. `search_github_repos(query: str)`: 搜索相关仓库并返回 Top 10（按相关性排序）。
2. `get_repo_overview(repo_name: str)`: 获取仓库的 README 摘要及根目录文件树。
3. `list_directory(repo_name: str, path: str = "")`: 获取指定目录下的文件和子文件夹列表，`path`为空字符串表示根目录。
4. `read_file_content(repo_name: str, file_path: str, start_line: int = 0, end_line: int = -1)`: 读取指定代码文件的特定行数内容，`end_line`为-1表示读取到文件尾。
5. `submit_findings(code_snippets: str = '', dependencies: str = '', summary: str = '')`: [终结动作] 当你找到了足够构建工具的代码或算法步骤，或者尝试了所有可能均宣告失败时，调用此工具结束任务。失败时不需要传入任何params。

# Workflow (Drill-Down 策略)
你必须遵循以下探索路径：
1. **探索 (Search):** 使用 `search_github_repos` 寻找高相关仓库。GitHub REST API不是一般搜索引擎，仓库通常不会将所有关键字全部写在名称和介绍里，你应当选择最核心的进行搜索（如查找CRT风格化时，搜索`crt filter`而不是`crt filter image processing`等）。
2. **侦察 (Recon):** 使用 `get_repo_overview` 查看仓库是否有价值。如果 README 描述不符，立即放弃并换一个仓库。
3. **下钻 (Drill):** 观察文件树，使用 `list_directory` 进入包含核心源码的目录（通常是 `src`, `core`, 或直接在根目录下的核心 `.py`/`.js` 文件）。
4. **提取 (Extract):** 使用 `read_file_content` 阅读目标文件。先阅读前 100 行（通常包含 import 和接口定义），确认是你要找的函数后，再精准拉取完整逻辑。
5. **总结 (Submit):** 整理出干净的代码片段或算法步骤，调用 `submit_findings` 提交。

# Strict Rules (绝对铁律)
1. **强制思考暂存 (Scratchpad):** 输出JSON中必须包含 `think` 字段，在其中记录：
   - 用户的原始目标是什么？
   - 我刚才看了什么？得到了什么结果？
   - 这个结果为什么有用/没用？
   - 我下一步打算调用哪个工具？为什么？
   - (可选，如果检查 **多个** 仓库后，发现确实无法在单一仓库得到全部功能) 该仓库实现某个功能的核心逻辑是什么？
2. **过滤噪音:** 绝对不要进入或读取以下目录和文件：`tests/`, `docs/`, `assets/`, `images/`, `node_modules/`, `.git/`, 配置文件（如 `.gitignore`, `package-lock.json` 等）。只关注核心源码。
3. **步数限制:** 你的探索必须高效。如果连续在 5 个不同的文件中都没有找到核心逻辑，必须立即放弃该仓库，去查看下一个仓库。
4. **切勿生造代码:** 你的任务是“寻找和搬运”，**绝对不要**自己编写业务逻辑代码或杜撰算法步骤。如果没找到，直接提交“未找到”。
5. **审查依赖:** 在阅读代码时，务必注意代码是否可以仅使用以下几个库实现（直接可运行或是改编后可运行）：`numpy`, `cv2` (opencv-contrib-python), `skimage` (scikit-image), `scipy`, `math`，它们是后续 ToolMakerAgent 成功运行的关键。
6. **跨语言参考:** **当且仅当** 多次尝试后依然无法找到完全匹配的代码（如：只有GLSL或其他语言而没有Python），允许对其他语言的仓库代码进行总结提炼。这种情况下，`code_snippets`提交的内容可以是原代码，也可以是伪代码或转写的Python。
7. **错误处理:** 如果工具报告网络错误等你无法修复的错误，直接提交“未找到”，在 `summary` 字段中说明遭遇了错误。
8. **见好就收:** 如果寻找多个仓库后仍有部分要求的功能无法实现，选择能够实现最多功能进行提交，不要因为贪心而超出步数限制。

# Search Guidance
A query can contain any combination of search qualifiers supported on GitHub. The format of the search query is:
`SEARCH_KEYWORD_1 SEARCH_KEYWORD_N QUALIFIER_1 QUALIFIER_N`

You cannot use queries that:
* Are longer than 256 characters (not including operators or qualifiers).
* Have more than five AND, OR, or NOT operators.

You can search for repositories on GitHub and narrow the results using these repository search qualifiers in any combination.
Use quotations around multi-word search terms. For example, if you want to search for issues with the label "In progress," you'd search for `label:"in progress"`. Search is not case sensitive.

## Search by repository name, description, or contents of the README file

With the `in` qualifier you can restrict your search to the repository name, repository description, repository topics, contents of the README file, or any combination of these. When you omit this qualifier, only the repository name, description, and topics are searched.

| Qualifier | Example |
| - | - |
| `in:name` | jquery in:name** matches repositories with "jquery" in the repository name. |
| `in:description` | **jquery in:name,description** matches repositories with "jquery" in the repository name or description. |
| `in:topics` | **jquery in:topics** matches repositories labeled with "jquery" as a topic. |
| `in:readme` | **jquery in:readme** matches repositories mentioning "jquery" in the repository's README file. |
| `repo:owner/name` | **repo:octocat/hello-world** matches a specific repository name. |

## Search based on the contents of a repository

You can find a repository by searching for content in the repository's README file using the `in:readme` qualifier.
Besides using `in:readme`, it's not possible to find repositories by searching for specific content within the repository. To search for a specific file or content within a repository, you can use the file finder or code-specific search qualifiers.

| Qualifier | Example                                                                                                                                                                |
| - | - |
| `in:readme` | **octocat in:readme** matches repositories mentioning "octocat" in the repository's README file. |

## Search by language

You can search repositories based on the language of the code in the repositories.

| Qualifier | Example |
| - | - |
| <code>language:<em>LANGUAGE</em></code> | **`rails language:javascript`** matches repositories with the word "rails" that are written in JavaScript. |

# Output Format Example
每一次回复，你必须输出使用Markdown包裹的严格的JSON格式，包含下列三个字段：
```json
{
    "think": "目标是寻找图片卡通化代码。我已经看了 Top 1 仓库 `cartoon-engine` 的 README，确认它符合要求。它根目录下有一个 `src/` 文件夹。我接下来需要调用 `list_directory` 查看 `src/` 里面的内容，寻找类似 `process.py` 或 `filter.py` 的文件。"
    "tool": "list_dirctory",
    "params": {
        "repo_name": "example/cartoon-engine",
        "path": "src"
    }
}
```
        """
        return prompt.strip()
    
    def generate_prompt(self, 
        user_intent: str = '', 
        tool_result: str = '', 
    ) -> str:
        user_prompt = ''
        if user_intent:
            user_prompt += f"# 历史\n{user_intent}"

        if len(self.query_cache) > 0:
            cached_query, cached_result = next(iter(self.query_cache.items()))
            if cached_query != 'No result' and cached_result.strip('\n ') != tool_result.strip('\n '):
                user_prompt += f"\n\n# 上次 search_github_repos('{cached_query}') 结果\n```\n{cached_result}\n```"

        if tool_result:
            user_prompt += f"\n\n# 工具调用结果\n```\n{tool_result}\n```"

        logger.info(f"注入提示词：\n{user_prompt}")
        return user_prompt

    def generate_code_stream(self, 
        user_intent: str = '', 
        tool_result: str = '', 
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
            user_intent, tool_result
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
        tool_result: str = '', 
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
            user_intent, tool_result
        ))
        
        # 提取并清洗代码块
        code = self._extract_json(llm_response)
        logger.info(f"成功生成代码：\n{code}")
        return code

    def execute(self, user_intent: str = '', tool_result: str = '') -> str:
        """
        实现父类BaseAgent的抽象execute方法，作为Agent对外的统一执行入口
        
        :param user_intent: 用户的图像增强意图
        :param init_details: 初始化基础量化信息
        :param previous_errors: 历史执行错误信息（用于代码修复）
        :return: 最终生成的可执行Python代码字符串
        """
        try:
            return self.generate_code(user_intent, tool_result)
        except Exception as e:
            logger.error(f"CoderAgent执行失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"编码Agent生成代码失败：{str(e)}") from e
        
    def execute_stream(
        self, 
        user_intent: str = '', 
        tool_result: str = '', 
    ) -> Generator[tuple[str, str], None, None]:
        """
        :return STREAM.REASONING: 流式返回思考内容
        :return STREAM.CONENT: 流式返回正文内容
        :return FINISH: 返回清理后的代码
        """
        try:
            for t, chunk in self.generate_code_stream(user_intent, tool_result):
                yield t, chunk
        except Exception as e:
            logger.error(f"CoderAgent执行失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"编码Agent生成代码失败：{str(e)}") from e