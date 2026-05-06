from agents.searcher import SearcherAgent

from github import Auth, Github
from typing import Generator, Literal

import time
import logging
import time
import json
import yaml

logger = logging.getLogger("Searcher")

class Searcher:
    def __init__(self, 
        github_token: str, 
        llm_client,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None = None,
        **kwargs
    ):
        auth = Auth.Token(github_token)
        self.github = Github(auth=auth)
        self.searcher = SearcherAgent(
            llm_client, model_name, 
            github_client=self.github, 
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **kwargs
        )

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

        rate_limit = self.github.get_rate_limit().resources
        if not rate_limit.search.remaining:
            yield 'SEARCH.API_LIMIT_REACHED', None
            return

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
            tool_result = self.use_tool(content['tool'], content['params']).strip(' \n')
            try:
                yield 'TOOL_CALL', {
                    'tool': content['tool'],
                    'params': f"```\n{json.dumps(content['params'], indent=2, ensure_ascii=False)}\n```",
                    'result': 
                        f"```\n{json.dumps(yaml.load(tool_result.strip('`'), yaml.FullLoader), indent=2, ensure_ascii=False)}\n```"
                        if content['tool'] != 'read_file_content' else f"```\n{tool_result.strip('`')}\n```"
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