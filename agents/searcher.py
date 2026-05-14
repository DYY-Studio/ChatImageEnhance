from agents.base_agent import BaseAgent
from agents.tools.web_search import WebSearchTool
from utils import get_executable_dir
from typing import Generator, Literal, Iterable
from collections import deque

from github import Github
from github.ContentFile import ContentFile

from modelscope.hub.api import HubApi
from modelscope.hub.file_download import model_file_download
from huggingface_hub import HfApi, RepoFile

import logging
import yaml
import httpx
import trafilatura

logger = logging.getLogger("SearcherAgent")

class SearcherAgent(BaseAgent):

    def __init__(self, 
        llm_client, 
        model_name: str = "gpt-4o-mini", 
        github_client: Github | None = None, 
        huggingface_token: str | None = None,
        modelscope_token: str | None = None,
        allowed_sources: Iterable[Literal["github", "huggingface", "modelscope"]] | None = None,
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
        normalized_allowed = {
            str(source).strip().lower() for source in (allowed_sources or ("github", "huggingface", "modelscope"))
            if str(source).strip()
        }
        normalized_allowed &= {"github", "huggingface", "modelscope"}
        if not normalized_allowed:
            normalized_allowed = {"github"}
        self.allowed_sources = normalized_allowed

        # 构造CoderAgent专属的系统提示词，明确代码生成规则
        system_prompt = self._build_system_prompt()
        # 调用父类初始化（LLM客户端、模型名、系统提示词、温度）
        super().__init__(llm_client, model_name, system_prompt, temperature, reasoning_effort, **kwargs)
        self.github_client = github_client
        self.modelscope_client = httpx.Client(
            base_url="https://modelscope.cn/openapi/v1", 
            headers={
                'Authorization': f"Bearer {modelscope_token}"
            } if modelscope_token is not None and modelscope_token else {}
        )
        self.modelscope_hubapi = HubApi(token=modelscope_token)
        self.hf_api = HfApi(token=huggingface_token if huggingface_token is not None and huggingface_token else None)
        if github_client is not None:
            github_client.per_page = 10
        self.curr_repo = None

        self.web_search_tool = WebSearchTool()
        self.result_cache: deque[str] = deque(maxlen=3)

        logger.info("CoderAgent 初始化完成，已加载全局算子注册表")

    def _search_web(self, query: str):
        response = self.web_search_tool.search(query, 10)
        if response.count > 0:
            yaml_text = yaml.dump([
                {
                    'title': result.title,
                    'snippet': result.snippet,
                    'url': result.url
                }
                for result in response.results
            ], allow_unicode=True, indent=2)
            return yaml_text
        else:
            return 'No result'

    def _read_html(self, url: str):
        res = self.web_search_tool.session.get(url)
        res.raise_for_status()
        return trafilatura.extract(res.text, output_format='markdown')

    def _read_localfile(self, path: str, start_line: int = 0, end_line: int = -1, clean_text: bool = False):
        try:
            lines = ''
            with open(path, mode='r', encoding='utf-8') as f:
                curr_line = -1
                for line in f:
                    curr_line += 1
                    if curr_line < start_line:
                        continue
                    if end_line > start_line and curr_line > end_line:
                        break
                    else:
                        lines += line
            return lines
        except Exception as e:
            return str(e)


    def _search_repos_github(self, query: str) -> str:
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
        return yaml_text if yaml_text != '[]' else 'No result'

    def _get_repo_overview_github(self, repo_name: str) -> str:
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

    def _list_directory_github(self, repo_name: str, path: str = "") -> str:
        if self.curr_repo is None or repo_name != self.curr_repo.full_name:
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

    def _read_file_github(self, repo_name: str, file_path: str, start_line: int = 0, end_line: int = -1) -> str:
        start_line = int(start_line)
        end_line = int(end_line)
        if self.curr_repo is None or repo_name != self.curr_repo.full_name:
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
        


    def _search_models_modelscope(self, query: str, task: str | None = None, library: str | None = None) -> str:
        params = {
            "search": str(query), 
        }
        if task is not None and task: params['filter_task'] = str(task)
        if library is not None and library: params['filter_library'] = str(library)

        try:
            res = self.modelscope_client.get(
                '/models',
                params=params
            )
            res.raise_for_status()
            res_json = res.json()

            if res_json['success']:
                if res_json['data']['total_count'] == 0:
                    return 'No result'
                return yaml.dump([{
                    "id": model['id'],
                    "name": model['display_name'],
                    "desc": model['description'],
                    "downloads": model['downloads'],
                    "likes": model['likes'],
                    "tasks": model['tasks'],
                    "params": model['params'],
                    "tags": model['tags']
                } for model in res_json['data']['models']], indent=2, allow_unicode=True)
            else:
                return 'Request failed'
        except Exception as e:
            import traceback
            traceback.print_exc()
            return str(e)
        
    def _list_directory_modelscope(self, model_id: str, path: str | None = None):
        files = self.modelscope_hubapi.get_model_files(model_id, root=path)
        if files:
            return yaml.dump([{
                "path": file['Path'],
                "type": file['Type'],
                "size": file['Size']
            } for file in files], indent=2, allow_unicode=True)
        else:
            return 'No result'

    def _read_file_modelscope(self, model_id: str, path: str, start_line: int = 0, end_line: int = -1):
        downloaded = model_file_download(model_id, path, local_dir=str(
            get_executable_dir() / "caches/modelscope"
        ))
        if downloaded:
            return self._read_localfile(downloaded, start_line, end_line)
        return f'Cannot download "{model_id}" file "{path}"'

    def _get_readme_modelscope(self, model_id: str, full: bool = False):
        readme = self.modelscope_hubapi.model_info(model_id).readme_content
        if readme:
            if len(readme) > 4096 and not full:
                return readme[:4096] + '... (over 4096 chars)'
            else: 
                return readme
        
        return 'Cannot get readme'
    


    def _search_models_hf(self, query: str, filter: str | None = None, pipeline_tag: str | None = None):
        models = list(self.hf_api.list_models(search=query, filter=filter, pipeline_tag=pipeline_tag, limit=10))
        if models:
            return yaml.dump([{
                "id": model.id,
                "library": model.library_name,
                "downloads": model.downloads if model.downloads_all_time is None else model.downloads_all_time,
                "likes": model.likes,
                "pipeline": model.pipeline_tag
            } for model in models])
    
        return 'No result'
    
    def _list_directory_hf(self, model_id: str, path: str | None = None):
        files = list(self.hf_api.list_repo_tree(model_id, path or ""))
        if not files:
            return 'No files'
        return yaml.dump([
            (
                {
                    "path": file.path,
                    "size": file.size,
                    "type": 'file'
                } if isinstance(file, RepoFile) else {
                    "path": file.path,
                    "type": 'dir'
                }
            )
            for file in files
        ], indent=2, allow_unicode=True)

    def _get_readme_hf(self, model_id: str, full: bool = False):
        if self.hf_api.file_exists(model_id, 'README.md'):
            readme = self._read_file_hf(model_id, 'README.md')
            if full or len(readme) < 4096:
                return readme
            else:
                return readme[:4096] + '... (over 4096 chars)'
        else:
            return 'No README.md'

    def _read_file_hf(self, model_id: str, path: str, start_line: int = 0, end_line: int = -1):
        downloaded = self.hf_api.hf_hub_download(
            model_id, path, local_dir=get_executable_dir() / "caches/hf"
        )
        if downloaded:
            return self._read_localfile(downloaded, start_line, end_line)
        return f'Cannot download "{model_id}" file "{path}"'



    def _submit_findings(
        self,
        repo_id: str = '',
        code_snippets: str = '', 
        dependencies: str = '', 
        summary: str = '', 
        source: Literal['github', 'huggingface', 'modelscope'] | None = None,
        require_files: list[str] | None = None
    ):
        return {
            "repo_id": repo_id,
            "code_snippets": code_snippets,
            "dependencies": dependencies,
            "summary": summary,
            "source": source,
            "require_files": require_files
        }

    def _build_system_prompt(self) -> str:
        """
        构建专属系统提示词，明确检索决策、来源偏好和结果提交规则
        核心原则：优先找到可下载、可运行、可维护的代码或模型推理方案
        """
        enabled_sources = sorted(self.allowed_sources)
        disabled_sources = [s for s in ("github", "huggingface", "modelscope") if s not in self.allowed_sources]
        source_policy = (
            f"可用检索源: {', '.join(enabled_sources)}。\n"
            + (
                f"禁用检索源: {', '.join(disabled_sources)}。你绝对不能调用这些来源对应的任何工具。"
                if disabled_sources else
                "当前没有禁用的检索源。"
            )
        )
        prompt = """
# Role: Code Searcher Agent
你是系统中负责“代码检索与评估”的资深研发工程师。
你的任务是利用工具搜索特定功能的实现代码或预训练模型推理脚本，经过层层筛选、阅读和评估，最终提取出高质量的核心逻辑代码片段或算法步骤，并移交给 ToolMakerAgent。
代码片段或算法步骤需要能够在 Python 中进行等价实现。

# Objective
根据用户的自然语言需求（例如：“寻找一段可以将图片卡通化的代码”或“找一个可以将图片动漫化的开源模型”），在 GitHub、Hugging Face 或 ModelScope 上定位最佳仓库/模型，提取可运行的核心算法或推理管道函数。
你的目标不是找到“看起来最有名”的项目，而是找到最适合当前环境、最容易下载权重、最容易封装成 Python 工具的方案。
运行时来源策略：
`web_search` / `read_html` 始终可用于理解任务名、算法名和模型名；$SOURCE_POLICY$

# Available Tools
你可以按信息增益选择工具，不需要机械执行所有步骤；一旦证据足够即可提交。
## Web Search Engine (寻找合适的算法的名称)
1. `web_search(query: str)`: 检索 DuckDuckGo (如果失败回落到 Bing)，返回前10条搜索结果
2. `read_html(url: str)`: 将指定URL指向的HTML文档转换为Markdown并返回

## GitHub (传统算法、综合代码库)
1. `search_repos_github(query: str)`: 搜索相关仓库并返回 Top 10（按相关性排序）。
2. `get_repo_overview_github(repo_name: str)`: 获取仓库的 README 摘要及根目录文件树。
3. `list_directory_github(repo_name: str, path: str = "")`: 获取指定目录下的文件和子文件夹列表，`path`为空字符串表示根目录。
4. `read_file_github(repo_name: str, file_path: str, start_line: int = 0, end_line: int = -1)`: 读取指定代码文件的特定行数内容，`end_line`为-1表示读取到文件尾。

## ModelScope (中国优质模型生态)
1. `search_models_modelscope(query: str, task: str | None = None, library: str | None = None)`: 搜索模型
2. `get_readme_modelscope(model_id: str, full: bool = False)`: 获取模型的 README 摘要或全文
3. `list_directory_modelscope(model_id: str, path: str | None = None)`: 获取指定目录下的文件和子文件夹列表
4. `read_file_modelscope(model_id: str, path: str, start_line: int = 0, end_line: int = -1)`: 读取模型库中的代码文件（如推理脚本）

## Hugging Face (国际主流模型生态)
1. `search_models_hf(query: str, filter: str | None = None, pipeline_tag: str | None = None)`: 搜索模型
2. `get_readme_hf(model_id: str, full: bool = False)`: 获取模型的 README 摘要或全文
3. `list_directory_hf(model_id: str, path: str | None = None)`: 获取指定目录下的文件和子文件夹列表
4. `read_file_hf(model_id: str, path: str, start_line: int = 0, end_line: int = -1)`: 读取模型库中的代码文件（如推理脚本）

## 终结动作
```
submit_findings(
    repo_id: str = '',
    code_snippets: str = '', 
    dependencies: str = '', 
    summary: str = '', 
    source: Optional[Literal['github', 'huggingface', 'modelscope']] = None,
    require_files: Optional[list[str]] = None
)
```
* 当你找到了足够构建工具的代码或算法步骤，或者尝试了所有可能均宣告失败时，调用此工具结束任务。
  - `repo_id`: 你找到的仓库或模型的ID（如`deepseek-ai/DeepSeek-R1`）
  - `code_snippets`: 
    - 对于传统代码，传入主要的处理逻辑即可。
    - 对于模型，还需要带上模型的载入逻辑（可以用`transformers`或`modelscope`直接`from_pretrained`吗？还是需要下载权重后自行加载？）。
    - 确保包含了必要的`import`语句
  - `dependencies`: 包名，用于从`pypi`安装，以空格分隔，不要传入其他内容。
  - `source`: 这个项目来自哪个源，可以填写`github`, `huggingface`或`modelscope`
  - `require_files`: 需要下载的文件路径列表（如权重、配置、tokenizer、processor）。GitHub 传统算法场景通常留空即可（留空不会下载任何文件）；HF/ModelScope 留空会触发来源默认下载策略。
* 失败时不需要传入任何params，（可选）或可以传入`summary`解释原因。

# Decision Policy (自适应检索决策)
每一步都先判断“下一次工具调用能否显著降低不确定性”。不要为了完成固定流程而浪费步骤。

1. **任务分类:**
   - 传统/确定性算法：直方图、CLAHE、Retinex、白平衡、锐化、去雾、边缘检测、形态学等，优先 GitHub 或 Web 找算法说明。
   - 深度学习/预训练模型：出现模型、权重、checkpoint、`from_pretrained`、diffusion、GAN、transformer、U-Net、超分、去噪、去模糊、分割、风格迁移、动漫化、卡通化、人像美化、抠图等信号时，优先 Hugging Face 和 ModelScope。
   - 混合/不确定任务：先用 Web 或 HF/ModelScope 的轻量搜索确认常见模型名，再决定是否需要 GitHub。
2. **来源优先级:**
   - 深度学习类任务默认顺序是 Hugging Face -> ModelScope -> GitHub。
   - 中文模型、国内可访问性、ModelScope pipeline 明确的任务，优先 ModelScope -> Hugging Face -> GitHub。
   - 国际通用模型、`diffusers` / `transformers` / `safetensors` 生态，优先 Hugging Face -> ModelScope -> GitHub。
   - GitHub 主要用于传统算法、官方模型代码补充、或 HF/ModelScope 没有推理说明时读取官方实现；不要把 GitHub 作为深度学习权重来源的首选。
3. **GitHub 深度学习项目降权规则:**
   - 如果 GitHub README 依赖 Google Drive、百度网盘、OneDrive、Release 大文件、Git LFS、手动下载 checkpoint，或权重链接不清晰，应立即降权或放弃。
   - 只有当 GitHub 项目明确提供可 pip 安装的推理包、官方 HF/ModelScope 模型 ID、或权重可由标准 Python API 自动下载时，才可选为最终方案。
   - 对 GitHub 深度学习项目，`source` 通常不应填 `github`，除非最终确实只依赖 GitHub 代码且不需要难下载权重。
4. **候选评分:**
   - 先看功能匹配度，再看权重下载可靠性，再看推理代码简洁度，再看依赖和设备适配，最后才看 stars/likes/downloads。
   - 优先选择 `from_pretrained`、`pipeline`、`snapshot_download`、`hf_hub_download`、`modelscope.pipeline` 可直接使用的模型。
   - 如果两个候选效果接近，选择依赖更少、权重更小、README 推理代码更明确、许可证更清晰的候选。

# Workflow (Adaptive Drill-Down 策略)
1. **定向 (Targeting):** 根据用户需求、设备信息和来源策略分类任务，明确为什么选择某个平台。
2. **探索 (Search):** 使用最匹配的平台搜索。深度学习类任务应优先尝试 `search_models_hf` 或 `search_models_modelscope`，必要时用 `web_search` 查模型别名或官方模型 ID。
3. **侦察 (Recon):** 使用 `get_readme` / `get_repo_overview` 判断是否真的能完成任务。对于模型重点检查 Usage、Inference、文件格式、权重下载方式和推荐依赖。
4. **下钻 (Drill):** 只在需要确认接口、依赖或最小文件清单时查看文件树和关键文件。HF/ModelScope 优先看 `README.md`, `app.py`, `inference.py`, `pipeline.py`, `config.json`, `model_index.json`；GitHub 优先看 `src`, `core`, `*.py`。
5. **提取 (Extract):** 提取可复用的最小推理逻辑或算法步骤，包含必要 import、模型加载、输入输出约定和设备选择。
6. **比较 (Compare):** 通常比较 2-3 个候选即可；如果第一个候选已经高度匹配且可下载可运行，可以见好就收。
7. **总结 (Submit):** 整理出干净的代码片段或算法步骤，正确填写 `source`, `repo_id`, `dependencies`, `require_files`。

# Strict Rules (绝对铁律)
1. **强制思考暂存 (Scratchpad):** 输出JSON中必须包含 `think` 字段，在其中记录：
   - 用户的原始目标是什么？用户是否指定了平台？
   - 我刚才看了什么？得到了什么结果？为什么有用/没用？
   - 下一步调用哪个工具？为什么？
   - 该代码/模型实现该功能的核心逻辑是什么？相比其他搜索结果有何优劣？
   - (可选) 尽可能精简的其他必要的跨步骤知识
2. **过滤噪音:** 绝对不要进入或读取 `tests/`, `docs/`, `assets/`, `.git/`、示例图片/视频等无关内容。  
   对于模型仓库，你可以查看文件树、README、推理脚本以及小型配置文件（如 `config.json`, `tokenizer_config.json`）来确定最小下载清单；不要读取大型权重文件内容本体。
3. **步数限制:** 你的探索必须高效。如果连续在 5 个不同的文件中都没有找到核心逻辑，必须立即放弃该仓库，去查看下一个仓库。
4. **切勿生造代码:** 你的任务是“寻找和搬运”，**绝对不要**杜撰不存在的模型调用 API 或算法逻辑。如果没找到，直接提交“未找到”。
5. **动态审查依赖:** 
   - 对于 **GitHub 传统算法**：务必注意代码是否可以仅使用 `numpy`, `cv2` (opencv-contrib-python), `skimage`, `math`, `PIL` 实现。
   - 对于 **Hugging Face / ModelScope 模型**：环境已经预装常用依赖 `torch`, `torchvision`, `transformers`, `diffusers`, `modelscope`。如果有其他必要的依赖，在 `submit_findings` 时必须准确列出。
6. **跨语言参考 (仅限GitHub):** 当且仅当多次尝试无法找到Python实现时，允许对其他语言的代码进行总结提炼，提交转写后的伪代码或Python代码。
7. **设备符合:** 参考传入的设备信息，选择能够在该设备上正常运行的实现，严禁选择参数量过大无法在设备上运行的项目。
8. **深度学习来源优先:** 深度学习类搜索必须优先尝试 Hugging Face / ModelScope。只有在二者没有合适模型、用户明确要求 GitHub、或 GitHub 是官方代码补充时，才使用 GitHub。
9. **最小下载优先:** 对于 GitHub / HuggingFace / ModelScope，优先提交最小 `require_files`，避免整仓下载。常见必需文件包括：权重文件、配置文件、tokenizer/processor 文件、推理必须脚本。若 HF/ModelScope 模型结构复杂且标准快照下载更可靠，`require_files` 可以留空并在 `summary` 中说明原因。
10. **权重可获得性优先:** 不要选择需要人工登录网页、网盘提取码、论坛下载、失效链接或 Git LFS 手动拉取的深度学习项目。可通过官方 Hub API 下载的候选优先级最高。
11. **错误处理:** 遇到网络或 API 错误等无法修复的问题，直接提交“未找到”，并在 `summary` 字段说明。
12. **见好就收:** 如果寻找多个目标后仍有部分功能无法实现，选择能实现最多功能的进行提交，不要因贪心超出步数限制。

# Search Guidance

## 1. GitHub 搜索技巧
* GitHub 优先用于传统图像处理算法、轻量工具库、官方代码参考。深度学习权重托管项目默认不优先。
* 使用 GitHub 支持的 Qualifier 进行精准搜索，格式为：`SEARCH_KEYWORD QUALIFIER`
    - `in:name` (按名称搜索, 如 `jquery in:name`)
    - `in:readme` (在README中搜索, 如 `cartoon in:readme`)
    - `language:python` (限定语言, 如 `style-transfer language:python`)
* 注意：搜索不要超过256个字符，不要使用过多逻辑运算符。

## 2. Hugging Face 搜索技巧
* Hugging Face 适合基于任务或管道寻找现代 AI 模型。
* **`pipeline_tag` (极其重要):** 准确指定任务类型能大幅提高搜索质量。对于图像任务，常用的 tags 包括：
  - `image-to-image`：图像到图像，如风格迁移、去噪
  - `image-segmentation`：图像分割，如抠图
  - `image-classification`：图像分类或质量判断
  - `text-to-image`：文生图或扩散模型相关参考
* **`filter`:** 可以限定框架（如 `pytorch`, `transformers`）或特定类别（如`image-to-image`）。
* 示例：寻找图像卡通化模型时，可以使用 `query="cartoon"`, `pipeline_tag="image-to-image"`。
* 如果 README 显示可用 `AutoModel.from_pretrained`, `DiffusionPipeline.from_pretrained`, `hf_hub_download`, `snapshot_download`，优先级提高。

## 3. ModelScope 搜索技巧
* ModelScope 中文生态极佳，适合寻找针对国内场景优化的模型。
* **`task` (极其重要):** 常用视觉任务包括：
  - `low-level-vision`：底层视觉，包括图像超分辨率、降噪等基础任务
    - `low-level-vision:image-super-resolution`
    - `low-level-vision:image-color-enhancement`
    - `low-level-vision:image-denoising`
    - `low-level-vision:image-deblurring`
    - `low-level-vision:image-demoireing`
    - `low-level-vision:image-debanding`
    - `low-level-vision:image-depth-estimation`
    - `low-level-vision:video-deinterlace`
    - `low-level-vision:video-super-resolution`
  - `vision-generation`: 视觉生成，包括风格迁移和卡通化等
    - `vision-generation:image-style-transfer`
    - `vision-generation:image-portrait-stylization`
  - `vision-editing`: 视觉编辑，包括图像上色、天空替换等
  - `vision-segmentation`: 视觉分割，包括抠图和目标分割等
* **`library`:** 可以限定使用的底层库（如 `pytorch`, `tf`）。
* 示例：寻找人像动漫化模型时，可以使用 `query="动漫"`, `task="vision-generation:image-portrait-stylization"`。
* 如果 README 显示可用 `modelscope.pipeline`、`snapshot_download` 或模型文件清单完整，优先级提高。

# Output Format Example
每一次回复，你必须输出使用Markdown包裹的严格的JSON格式，包含下列四个字段：
```json
{
    "think": "目标是寻找图片卡通化代码。我已经看了 Top 1 仓库 `cartoon-engine` 的 README，确认它符合要求。它根目录下有一个 `src/` 文件夹。我接下来需要调用 `list_directory` 查看 `src/` 里面的内容，寻找类似 `process.py` 或 `filter.py` 的文件。",
    "cache": false, // 要求流程控制系统缓存上次调用结果，下次的提示词会携带已缓存的内容。最多缓存3个结果，此后会删除最旧缓存再增加新缓存。推荐缓存：搜索结果列表，关键文件，文件列表。
    "tool": "list_directory_github",
    "params": {
        "repo_name": "example/cartoon-engine",
        "path": "src"
    }
}
```
        """
        return prompt.strip().replace("$SOURCE_POLICY$", source_policy, 1)
    
    def generate_prompt(self, 
        user_intent: str = '', 
        tool_result: str = '', 
    ) -> str:
        user_prompt = ''
        if user_intent:
            user_prompt += f"# 历史\n{user_intent}"

        if tool_result:
            user_prompt += f"\n\n# 工具调用结果\n```\n{tool_result}\n```"

        if not self.result_cache:
            for idx in range(len(self.result_cache)):
                cache_str = self.result_cache.popleft()
                user_prompt += f'\n# 结果缓存{idx+1}\n```\n{cache_str}\n```'
                self.result_cache.append(cache_str)

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
