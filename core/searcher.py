from agents.searcher import SearcherAgent

from github import Auth, Github
from typing import Generator, Literal

import logging
import time

logger = logging.getLogger("Searcher")

class Searcher:
    def __init__(self, github_token: str, llm_client, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        auth = Auth.Token(github_token)
        self.github = Github(auth=auth)
        self.searcher = SearcherAgent(llm_client, model_name, github_client=self.github, temperature=temperature)

    def use_tool(self, tool_name: str, params: dict):
        if (func := getattr(self.searcher, f"_{tool_name}", False)):
            try:
                return func(**params)
            except Exception as e:
                return f"Tool Use Error: {e}"
        else:
            return f"Tool Use Error: Cannot find tool {tool_name}"

    def search(self, prompt: str, steps_limit: int = 30) -> Generator[tuple[str, str], None, None]:
        content: dict | None = None
        thinks: str = f"用户输入: {prompt.strip()}"
        tool_result: str = ''

        times: int = 0
        while content is None or 'tool' not in content or content['tool'] != 'submit_findings':
            times += 1
            for t, chunk in self.searcher.execute_stream(thinks + tool_result):
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
                yield 'SEARCH.LIMIT_REACHED', None
            
            tool_result = self.use_tool(content['tool'], content['params']).strip(' \n')
            tool_result = f"\n\n工具调用结果:\n```\n{tool_result}\n```"
            logger.info(tool_result.strip('\n'))

            time.sleep(0.2)

        yield 'SEARCH.FINISH', content['params'] if content and 'params' in content else None