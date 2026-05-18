import unittest

from core.searcher import Searcher


class _NoRateLimitGithub:
    def get_rate_limit(self):
        raise RuntimeError("rate limit unavailable in test")


class _FakeSearcherAgent:
    def __init__(self):
        self.calls = []

    def execute_stream(self, thinks: str = "", tool_result: str = ""):
        self.calls.append((thinks, tool_result))
        if len(self.calls) == 1:
            yield "FINISH", {
                "think": "try a model with a missing weight file",
                "tool": "submit_findings",
                "params": {
                    "source": "huggingface",
                    "repo_id": "bad/model",
                    "require_files": ["missing.safetensors"],
                    "summary": "bad candidate",
                    "code_snippets": "load bad/model",
                },
            }
            return

        yield "FINISH", {
            "think": "switch to a downloadable model",
            "tool": "submit_findings",
            "params": {
                "source": "huggingface",
                "repo_id": "good/model",
                "require_files": ["model.safetensors"],
                "summary": "good candidate",
                "code_snippets": "load good/model",
            },
        }


class _FeedbackSearcher(Searcher):
    def __init__(self):
        self.allowed_sources = {"huggingface"}
        self.github = _NoRateLimitGithub()
        self.searcher = _FakeSearcherAgent()
        self.enrich_calls = []

    def enrich_findings(self, findings, auto_download=True, progress_callback=None):
        self.enrich_calls.append(dict(findings or {}))
        enriched = dict(findings or {})
        enriched.setdefault("additional_packages", [])
        enriched.setdefault("additional_imports", [])
        enriched.setdefault("downloaded_files", [])
        enriched.setdefault("download_dir", None)
        enriched["download_attempted"] = bool(auto_download)
        if enriched.get("repo_id") == "bad/model":
            enriched["download_error"] = "404 Client Error: missing.safetensors"
            enriched["download_error_info"] = {
                "source": enriched.get("source"),
                "repo_id": enriched.get("repo_id"),
                "failed_file": "missing.safetensors",
                "failed_url": None,
                "failed_filename": None,
                "stage": "huggingface_require_file",
                "submitted_require_files": enriched.get("require_files"),
                "submitted_asset_urls": enriched.get("asset_urls"),
                "require_files": enriched.get("require_files"),
                "asset_urls": enriched.get("asset_urls"),
                "error_type": "HTTPStatusError",
                "error": enriched["download_error"],
                "message": enriched["download_error"],
            }
        else:
            enriched["download_error"] = None
            enriched["download_error_info"] = None
            enriched["downloaded_files"] = ["cache/good/model.safetensors"]
            enriched["download_dir"] = "cache/good"
        return enriched


class _FailingDownloadSearcher(Searcher):
    def __init__(self):
        self.allowed_sources = {"huggingface"}

    def _download_hf_assets(self, repo_id, require_files):
        raise RuntimeError("cannot fetch model.safetensors")


class _FailingHfApi:
    def hf_hub_download(self, repo_id, filename, local_dir):
        raise RuntimeError(f"cannot fetch {filename} from {repo_id}")


class _HfFileFailureSearcher(Searcher):
    def __init__(self):
        self.allowed_sources = {"huggingface"}
        self.searcher = type("FakeAgent", (), {"hf_api": _FailingHfApi()})()


class _UrlFailureSearcher(Searcher):
    def __init__(self):
        self.allowed_sources = {"github"}

    def _download_url_asset(self, asset, cache_dir, progress_callback=None, index=1, total=1):
        raise RuntimeError(f"cannot fetch direct url {asset['url']}")


class SearcherDownloadFeedbackTests(unittest.TestCase):
    def test_search_retries_when_submit_findings_download_fails(self):
        searcher = _FeedbackSearcher()

        events = list(searcher.search(
            "need a model",
            steps_limit=4,
            interval=0,
            auto_download_findings=True,
        ))

        finish_payloads = [body for event, body in events if event == "SEARCH.FINISH"]
        self.assertEqual(len(finish_payloads), 1)
        self.assertEqual(finish_payloads[0]["repo_id"], "good/model")
        self.assertEqual(finish_payloads[0]["download_error"], None)
        self.assertEqual(finish_payloads[0]["downloaded_files"], ["cache/good/model.safetensors"])

        self.assertEqual(len(searcher.searcher.calls), 2)
        _, retry_tool_result = searcher.searcher.calls[1]
        self.assertIn("Model asset download failed after submit_findings", retry_tool_result)
        self.assertIn("bad/model", retry_tool_result)
        self.assertIn("missing.safetensors", retry_tool_result)
        self.assertIn("submitted_require_files", retry_tool_result)
        self.assertIn("failed_file", retry_tool_result)

        self.assertTrue(any(event.startswith("SEARCH.DOWNLOAD_ERROR") for event, _ in events))

    def test_enrich_findings_records_structured_download_failure(self):
        searcher = _FailingDownloadSearcher()

        enriched = searcher.enrich_findings(
            {
                "source": "huggingface",
                "repo_id": "bad/model",
                "require_files": ["model.safetensors"],
            },
            auto_download=True,
        )

        self.assertEqual(enriched["download_error"], "cannot fetch model.safetensors")
        self.assertEqual(enriched["download_error_info"]["source"], "huggingface")
        self.assertEqual(enriched["download_error_info"]["repo_id"], "bad/model")
        self.assertEqual(enriched["download_error_info"]["require_files"], ["model.safetensors"])
        self.assertEqual(enriched["download_error_info"]["error_type"], "RuntimeError")

    def test_enrich_findings_identifies_failed_required_file(self):
        searcher = _HfFileFailureSearcher()

        enriched = searcher.enrich_findings(
            {
                "source": "huggingface",
                "repo_id": "bad/model",
                "require_files": ["model.safetensors", "config.json"],
            },
            auto_download=True,
        )

        info = enriched["download_error_info"]
        self.assertEqual(info["repo_id"], "bad/model")
        self.assertEqual(info["failed_file"], "model.safetensors")
        self.assertIsNone(info["failed_url"])
        self.assertEqual(info["stage"], "huggingface_require_file")
        self.assertEqual(info["submitted_require_files"], ["model.safetensors", "config.json"])
        self.assertEqual(info["submitted_asset_urls"], [])
        self.assertIn("model.safetensors", info["error"])

    def test_enrich_findings_identifies_failed_asset_url(self):
        searcher = _UrlFailureSearcher()

        enriched = searcher.enrich_findings(
            {
                "source": "github",
                "repo_id": "owner/repo",
                "asset_urls": [
                    {
                        "url": "https://example.com/releases/model.pth",
                        "filename": "model.pth",
                    }
                ],
            },
            auto_download=True,
        )

        info = enriched["download_error_info"]
        self.assertEqual(info["repo_id"], "owner/repo")
        self.assertIsNone(info["failed_file"])
        self.assertEqual(info["failed_url"], "https://example.com/releases/model.pth")
        self.assertEqual(info["failed_filename"], "model.pth")
        self.assertEqual(info["stage"], "direct_url_asset")
        self.assertEqual(info["submitted_require_files"], [])
        self.assertEqual(
            info["submitted_asset_urls"],
            [{"url": "https://example.com/releases/model.pth", "filename": "model.pth"}],
        )


if __name__ == "__main__":
    unittest.main()
