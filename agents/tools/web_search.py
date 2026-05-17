import re
import html
import base64
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
import httpx

'''
Reference:
https://github.com/Hmbown/DeepSeek-TUI/blob/main/crates/tui/src/tools/web_search.rs
'''

@dataclass
class WebSearchEntry:
    title: str
    url: str
    snippet: Optional[str]

@dataclass
class WebSearchResponse:
    query: str
    source: str
    count: int
    message: str
    results: List[WebSearchEntry]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# DDG
TITLE_RE = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>')
SNIPPET_RE = re.compile(
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>'
)

# Bing
BING_RESULT_RE = re.compile(r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', re.I | re.S)
BING_TITLE_RE = re.compile(r'<h2[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
BING_SNIPPET_RE = re.compile(r'<div[^>]*class="[^"]*\bb_caption\b[^"]*"[^>]*>.*?<p[^>]*>(.*?)</p>', re.I | re.S)

# 移除 HTML 标签
TAG_RE = re.compile(r'<[^>]+>')

# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------
def normalize_text(text: str) -> str:
    """清理 HTML 标签、解码实体并压缩空白字符"""
    stripped = TAG_RE.sub('', text)
    decoded = html.unescape(stripped) # 完美替代 Rust 的 decode_html_entities
    return " ".join(decoded.split())

def extract_query_param(url: str, key: str) -> Optional[str]:
    """从 URL 中提取指定的 query 参数"""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return qs.get(key, [None])[0]

def normalize_ddg_url(href: str) -> str:
    """标准化 DuckDuckGo 的跳转 URL"""
    uddg = extract_query_param(href, "uddg")
    if uddg:
        decoded = unquote(uddg)
        if decoded: return decoded
    if href.startswith('//'):
        return f"https:{href}"
    if href.startswith('/'):
        return f"https://duckduckgo.com{href}"
    return href

def normalize_bing_url(href: str) -> str:
    """标准化 Bing 的加密跳转 URL"""
    u_param = extract_query_param(href, "u")
    if u_param:
        decoded = unquote(u_param)
        token = decoded[2:] if decoded.startswith("a1") else decoded
        padded = token.replace('-', '+').replace('_', '/')
        # 补齐 Base64 padding
        padded += "=" * ((4 - len(padded) % 4) % 4)
        try:
            url = base64.b64decode(padded).decode('utf-8')
            if url.startswith("http://") or url.startswith("https://"):
                return url
        except Exception:
            pass
            
    if href.startswith('//'):
        return f"https:{href}"
    if href.startswith('/'):
        return f"https://www.bing.com{href}"
    return href

def root_domain(url_str: str) -> Optional[str]:
    """提取 URL 的根域名 (eTLD+1 的简易替代方案，用于 Spam 过滤)"""
    try:
        after_scheme = url_str.split('://', 1)[-1] if '://' in url_str else url_str
        host = after_scheme.split('/')[0].split('?')[0].split('#')[0]
        host = host.split('@')[-1].split(':')[0].lower()
        if not host:
            return None
        labels = [lbl for lbl in host.split('.') if lbl]
        if len(labels) <= 2:
            return host
        return ".".join(labels[-2:])
    except Exception:
        return None

def is_likely_spam_results(results: List[WebSearchEntry]) -> bool:
    """检测搜索结果是否为单域名霸屏的垃圾页面 (>= 60%)"""
    if len(results) < 3:
        return False
        
    counts = {}
    for r in results:
        domain = root_domain(r.url)
        if domain:
            counts[domain] = counts.get(domain, 0) + 1
            
    if not counts:
        return False
        
    max_count = max(counts.values())
    return max_count * 5 >= len(results) * 3

# ---------------------------------------------------------
# 核心搜索逻辑类
# ---------------------------------------------------------
class WebSearchTool:
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
    DEFAULT_MAX_RESULTS = 5
    DEFAULT_TIMEOUT_MS = 15000

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.timeout = timeout_ms / 1000.0
        self.session = httpx.Client()
        self.session.headers.update({"User-Agent": self.USER_AGENT})

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> WebSearchResponse:
        """执行搜索：默认 DDG，如果失败或遇到反爬验证码则回退至 Bing"""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
            
        max_results = max(1, min(max_results, 10))  # clamp(1, 10)
        
        # 1. 尝试 DuckDuckGo 搜索
        encoded_query = quote_plus(query)
        ddg_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        try:
            resp = self.session.get(
                ddg_url, 
                headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", 
                         "Accept-Language": "en-US,en;q=0.5"},
                timeout=self.timeout
            )
            resp.raise_for_status()
            html_body = resp.text
            
            results = self._parse_duckduckgo_results(html_body, max_results)
            source = "duckduckgo"
            message_suffix = None
            
            # 若无结果，检查是否触发反爬或解析异常，触发 Bing 回退
            if not results:
                is_challenge = "anomaly-modal" in html_body or "Unfortunately, bots use DuckDuckGo too" in html_body
                
                fallback_results = self._run_bing_search(query, max_results)
                if fallback_results:
                    results = fallback_results
                    source = "bing"
                    message_suffix = "DuckDuckGo returned a bot challenge; used Bing fallback" if is_challenge \
                        else "DuckDuckGo returned no parseable results; used Bing fallback"
                elif is_challenge:
                    raise Exception("DuckDuckGo returned a bot challenge and Bing fallback returned no results")
                    
        except Exception as e:
            # 请求失败时也触发 Bing 回退
            fallback_results = self._run_bing_search(query, max_results)
            if fallback_results:
                results = fallback_results
                source = "bing"
                message_suffix = f"DuckDuckGo request failed ({e}); used Bing fallback"
            else:
                raise Exception(f"DuckDuckGo request failed and Bing fallback returned no results: {e}")

        # 整理返回信息
        if not results:
            message = "No results found"
        elif message_suffix:
            message = f"Found {len(results)} result(s). {message_suffix}"
        else:
            message = f"Found {len(results)} result(s)"

        return WebSearchResponse(
            query=query,
            source=source,
            count=len(results),
            message=message,
            results=results
        )

    def _run_bing_search(self, query: str, max_results: int) -> List[WebSearchEntry]:
        encoded_query = quote_plus(query)
        bing_url = f"https://www.bing.com/search?q={encoded_query}"
        
        try:
            resp = self.session.get(
                bing_url,
                headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                         "Accept-Language": "en-US,en;q=0.9"},
                timeout=self.timeout,
                follow_redirects=True
            )
            resp.raise_for_status()
            return self._parse_bing_results(resp.text, max_results)
        except Exception:
            return []

    def _parse_duckduckgo_results(self, html_body: str, max_results: int) -> List[WebSearchEntry]:
        snippets = []
        for match in SNIPPET_RE.finditer(html_body):
            snippet_raw = match.group(1) or match.group(2)
            if snippet_raw:
                snippets.append(normalize_text(snippet_raw))

        results = []
        for idx, match in enumerate(TITLE_RE.finditer(html_body)):
            if len(results) >= max_results:
                break
                
            href = match.group(1)
            title_raw = match.group(2)
            title = normalize_text(title_raw)
            if not title:
                continue
                
            url = normalize_ddg_url(href)
            snippet = snippets[idx] if idx < len(snippets) else None
            if snippet and not snippet.strip():
                snippet = None

            results.append(WebSearchEntry(title=title, url=url, snippet=snippet))

        if is_likely_spam_results(results):
            return []
        return results

    def _parse_bing_results(self, html_body: str, max_results: int) -> List[WebSearchEntry]:
        results = []
        for block_match in BING_RESULT_RE.finditer(html_body):
            if len(results) >= max_results:
                break
                
            block = block_match.group(1)
            title_match = BING_TITLE_RE.search(block)
            if not title_match:
                continue
                
            href = title_match.group(1)
            title_raw = title_match.group(2)
            title = normalize_text(title_raw)
            if not title:
                continue

            snippet = None
            snippet_match = BING_SNIPPET_RE.search(block)
            if snippet_match:
                snippet = normalize_text(snippet_match.group(1))
                if not snippet.strip():
                    snippet = None

            url = normalize_bing_url(href)
            results.append(WebSearchEntry(title=title, url=url, snippet=snippet))

        if is_likely_spam_results(results):
            return []
        return results