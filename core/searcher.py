from agents.searcher import SearcherAgent

from github import Auth, Github
from typing import Generator, Literal

import time
import logging
import json
import yaml
import os
import re
import importlib.metadata as importlib_metadata
from pathlib import Path

from utils import get_executable_dir

logger = logging.getLogger("Searcher")

class Searcher:
    # 近似 from_pretrained 的“最小必需文件”过滤规则：
    # 保留权重 + 配置 + tokenizer/processor + 推理相关代码；忽略样例图、文档、测试等噪声内容。
    _MODEL_ALLOW_PATTERNS: list[str] = [
        "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.tflite", "*.gguf",
        "*.json", "*.yaml", "*.yml", "*.txt", "*.model", "*.spm", "*.bpe", "*.jinja",
        "*.py",
        "config.json", "generation_config.json", "model_index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "vocab.txt", "merges.txt",
        "preprocessor_config.json", "processor_config.json", "feature_extractor_config.json",
        "unet/*.json", "vae/*.json", "text_encoder/*.json", "text_encoder_2/*.json",
        "scheduler/*.json"
    ]
    _MODEL_IGNORE_PATTERNS: list[str] = [
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp", "*.svg",
        "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm",
        "*.wav", "*.mp3", "*.flac", "*.ogg",
        "*.md", "*.rst", "*.pdf",
        "assets/**", "figures/**", "images/**", "media/**",
        "docs/**", "doc/**",
        "demo/**", "demos/**", "example/**", "examples/**", "samples/**", "sample/**",
        "tests/**", "test/**", "benchmark/**", "benchmarks/**",
        "training/**", "train/**"
    ]
    # 常见“分发包名 != import 模块名”映射
    _PKG_IMPORT_ALIASES: dict[str, str] = {
        "pillow": "PIL",
        "opencv-python": "cv2",
        "opencv-contrib-python": "cv2",
        "opencv-python-headless": "cv2",
        "scikit-image": "skimage",
        "pyyaml": "yaml",
        "huggingface-hub": "huggingface_hub",
        "python-dateutil": "dateutil",
        "beautifulsoup4": "bs4",
        "faiss-cpu": "faiss",
        "faiss-gpu": "faiss",
        "pytorch-lightning": "pytorch_lightning",
        "sentence-transformers": "sentence_transformers",
    }

    def __init__(self, 
        llm_client,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None,
        github_token: str | None = None,
        huggingface_token: str | None = None,
        modelscope_token: str | None = None,
        **kwargs
    ):
        if github_token:
            auth = Auth.Token(github_token)
            self.github = Github(auth=auth)
        else:
            self.github = Github()
        self.searcher = SearcherAgent(
            llm_client, model_name, 
            github_client=self.github, 
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            huggingface_token=huggingface_token,
            modelscope_token=modelscope_token,
            **kwargs
        )

    @staticmethod
    def _canonical_dist_name(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name.strip().lower())

    @staticmethod
    def _extract_dist_name(requirement: str) -> str:
        req = str(requirement or "").strip()
        if not req:
            return ""
        req = req.split(";", maxsplit=1)[0].strip()
        req = req.split("[", maxsplit=1)[0].strip()
        req = re.split(r"(==|!=|>=|<=|>|<|~=)", req, maxsplit=1)[0].strip()
        return req

    @classmethod
    def _resolve_import_from_installed_dist(cls, dist_name: str) -> str | None:
        if not dist_name:
            return None
        # 优先读取 top_level.txt（最可靠）
        try:
            dist = importlib_metadata.distribution(dist_name)
            top_level = dist.read_text("top_level.txt") or ""
            for line in top_level.splitlines():
                mod = line.strip()
                if mod and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", mod):
                    return mod
        except Exception:
            pass

        # 回退：从 packages_distributions 反向匹配
        try:
            canonical_target = cls._canonical_dist_name(dist_name)
            for mod, dists in importlib_metadata.packages_distributions().items():
                if any(cls._canonical_dist_name(d) == canonical_target for d in (dists or [])):
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", mod):
                        return mod
        except Exception:
            pass
        return None

    @staticmethod
    def _split_dependencies(dependencies: str | None) -> list[str]:
        if not dependencies:
            return []

        raw_tokens = re.split(r"[\s,]+", str(dependencies).strip())
        sanitized: list[str] = []
        for token in raw_tokens:
            if not token:
                continue
            token = token.strip().strip("'\"")
            if not token:
                continue
            # 仅允许合法 pip 依赖格式，避免注入或脏输入
            if re.fullmatch(
                r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.\-,]+\])?(?:\s*(?:==|!=|>=|<=|>|<|~=)\s*[A-Za-z0-9*+_.-]+)?",
                token
            ):
                sanitized.append(token.replace(" ", ""))
        return sanitized

    @classmethod
    def _dependency_to_import(cls, dep: str) -> str | None:
        dist_name = cls._extract_dist_name(dep)
        if not dist_name:
            return None

        canonical = cls._canonical_dist_name(dist_name)
        if canonical in cls._PKG_IMPORT_ALIASES:
            return cls._PKG_IMPORT_ALIASES[canonical]

        resolved = cls._resolve_import_from_installed_dist(dist_name)
        if resolved:
            return resolved

        # 回退策略：按常规转换
        name = dist_name.replace("-", "_").split(".", maxsplit=1)[0]
        return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else None

    @staticmethod
    def _normalize_require_files(require_files: list[str] | None) -> list[str]:
        if not isinstance(require_files, list):
            return []
        normalized: list[str] = []
        for item in require_files:
            if not isinstance(item, str):
                continue
            path = item.strip().replace("\\", "/").strip("/")
            if not path or ".." in path.split("/"):
                continue
            normalized.append(path)
        return normalized

    @staticmethod
    def _get_repo_cache_dir(source: str, repo_id: str) -> Path:
        safe_repo = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id.strip())
        return get_executable_dir() / "caches" / "model_assets" / source / safe_repo

    def _download_hf_assets(self, repo_id: str, require_files: list[str]) -> tuple[list[str], str]:
        cache_dir = self._get_repo_cache_dir("huggingface", repo_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files: list[str] = []
        if require_files:
            for rel_path in require_files:
                downloaded = self.searcher.hf_api.hf_hub_download(
                    repo_id=repo_id,
                    filename=rel_path,
                    local_dir=str(cache_dir)
                )
                downloaded_files.append(str(Path(downloaded).resolve()))
            return downloaded_files, str(cache_dir.resolve())

        # 未指定具体文件时，下载整个仓库快照
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            endpoint=os.environ.get("HF_ENDPOINT"),
            allow_patterns=self._MODEL_ALLOW_PATTERNS,
            ignore_patterns=self._MODEL_IGNORE_PATTERNS
        )
        downloaded_files.append(str(Path(snapshot_dir).resolve()))
        return downloaded_files, str(cache_dir.resolve())

    def _download_modelscope_assets(self, repo_id: str, require_files: list[str]) -> tuple[list[str], str]:
        cache_dir = self._get_repo_cache_dir("modelscope", repo_id)
        cache_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files: list[str] = []
        if require_files:
            from modelscope.hub.file_download import model_file_download
            for rel_path in require_files:
                downloaded = model_file_download(
                    repo_id,
                    rel_path,
                    local_dir=str(cache_dir)
                )
                downloaded_files.append(str(Path(downloaded).resolve()))
            return downloaded_files, str(cache_dir.resolve())

        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
        try:
            snapshot_dir = ms_snapshot_download(
                repo_id,
                local_dir=str(cache_dir),
                allow_patterns=self._MODEL_ALLOW_PATTERNS,
                ignore_patterns=self._MODEL_IGNORE_PATTERNS
            )
        except TypeError:
            # 向后兼容旧版参数名
            snapshot_dir = ms_snapshot_download(
                repo_id,
                cache_dir=str(cache_dir),
                allow_file_pattern=self._MODEL_ALLOW_PATTERNS,
                ignore_file_pattern=self._MODEL_IGNORE_PATTERNS
            )
        downloaded_files.append(str(Path(snapshot_dir).resolve()))
        return downloaded_files, str(cache_dir.resolve())

    def enrich_findings(self, findings: dict | None, auto_download: bool = True) -> dict:
        """
        规范化 SearcherAgent 的 submit_findings 结果，并可选自动下载模型资产。
        """
        if not isinstance(findings, dict):
            return {}

        enriched = dict(findings)
        deps = self._split_dependencies(enriched.get("dependencies"))
        import_names = [name for name in (self._dependency_to_import(dep) for dep in deps) if name]

        # 保序去重
        dedup_imports: list[str] = []
        for name in import_names:
            if name not in dedup_imports:
                dedup_imports.append(name)

        enriched["additional_packages"] = deps
        enriched["additional_imports"] = dedup_imports

        source = enriched.get("source")
        repo_id = str(enriched.get("repo_id") or "").strip()
        require_files = self._normalize_require_files(enriched.get("require_files"))

        enriched["downloaded_files"] = []
        enriched["download_dir"] = None
        enriched["download_error"] = None
        enriched["require_files"] = require_files

        if not auto_download or source not in ("huggingface", "modelscope") or not repo_id:
            return enriched

        try:
            if source == "huggingface":
                files, folder = self._download_hf_assets(repo_id, require_files)
            else:
                files, folder = self._download_modelscope_assets(repo_id, require_files)
            enriched["downloaded_files"] = files
            enriched["download_dir"] = folder
        except Exception as e:
            enriched["download_error"] = str(e)

        return enriched

    def use_tool(self, tool_name: str, params: dict):
        if (func := getattr(self.searcher, f"_{tool_name}", False)):
            try:
                return func(**params)
            except Exception as e:
                return f"Tool Use Error: {e}"
        else:
            return f"Tool Use Error: Cannot find tool {tool_name}"

    def search(self, prompt: str, steps_limit: int = 30, interval: float = 0.5) -> Generator[tuple[str, str | dict], None, None]:
        content: dict | None = None
        thinks: str = f"用户输入: {prompt.strip()}\n用户限制：最多执行{steps_limit}次"
        tool_result: str = ''

        github_search_available = True
        try:
            rate_limit = self.github.get_rate_limit().resources
            github_search_available = bool(rate_limit.search.remaining)
            if not github_search_available:
                thinks += "\n系统限制：GitHub Search API额度已耗尽，请避免使用GitHub工具，优先使用HuggingFace或ModelScope工具。"
                yield 'SEARCH.API_LIMIT_REACHED', None
        except Exception:
            pass

        times: int = 0
        while content is None or 'tool' not in content or content['tool'] != 'submit_findings':
            times += 1
            start_time = time.perf_counter()
            for t, chunk in self.searcher.execute_stream(thinks, tool_result):
                if t == "STREAM.REASONING":
                    yield f"SEARCH.REASONING.{times}", chunk
                elif t == "STREAM.CONTENT":
                    yield f"SEARCH.CONTENT.{times}", chunk
                elif t == "FINISH":
                    content = chunk
                    yield f"SEARCH.STEP.FINISH", content['think']

            thinks += f"\n运行{times}: {content['think']}"
            yield f"THINK.{times}", content['think']
            logger.info(f"运行{times}: {content['think']}")
            if content['tool'] == 'submit_findings':
                break

            if times >= steps_limit:
                yield 'SEARCH.STEPS_LIMIT_REACHED', None
                return
            
            thinks += f"调用{times}: {content['tool']}, {json.dumps(content['params'], ensure_ascii=False)}"
            if (not github_search_available) and content['tool'].endswith("_github"):
                tool_result = "Tool Use Error: GitHub Search API limit reached"
            else:
                tool_result = f"{content['tool']}, {json.dumps(content['params'], ensure_ascii=False)}\n" + str(self.use_tool(content['tool'], content['params'])).strip(' \n')
            try:
                yield 'TOOL_CALL', {
                    'tool': content['tool'],
                    'params': f"```\n{json.dumps(content['params'], indent=2, ensure_ascii=False)}\n```",
                    'result': 
                        f"```\n{json.dumps(yaml.load(tool_result.strip('`'), yaml.FullLoader), indent=2, ensure_ascii=False)}\n```"
                        if not content['tool'].startswith('read_file_') else f"```\n{tool_result.strip('`')}\n```"
                }
            except:
                yield 'TOOL_CALL', {
                    'tool': content['tool'],
                    'params': f"```\n{json.dumps(content['params'], indent=2, ensure_ascii=False)}\n```",
                    'result': f"```\n{tool_result.strip('`')}\n```"
                }
            logger.info(tool_result.strip('\n'))

            if (duration := (time.perf_counter() - start_time)) < interval:
                time.sleep(interval - duration)

        yield 'SEARCH.FINISH', content['params'] if content and 'params' in content else None
