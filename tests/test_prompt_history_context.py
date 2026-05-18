import unittest
from types import SimpleNamespace
from unittest.mock import patch

import components


class PromptHistoryContextTests(unittest.TestCase):
    def test_attempt_history_summary_records_operators_and_following_feedback(self):
        messages = [
            {"role": "user", "content": "提升暗部细节"},
            {
                "role": "assistant",
                "content": "round 1",
                "image": object(),
                "process_code": "def process(img, trial, cache):\n    return cv_wrappers.CLAHE_Enhancement(img)",
            },
            {"role": "user", "content": "噪声太明显"},
            {
                "role": "assistant",
                "content": "round 2",
                "image": object(),
                "process_code": "def process(img, trial, cache):\n    return cv_wrappers.Bilateral_Filter(img)",
            },
        ]

        summary = components._build_attempt_history_summary(messages, last_assistant_index=3)

        self.assertIn("CLAHE_Enhancement", summary)
        self.assertIn("噪声太明显", summary)
        self.assertNotIn("Bilateral_Filter", summary)

    def test_generate_user_prompt_keeps_previous_code_and_summarizes_older_rounds(self):
        older_process = """
def process(img, trial, cache):
    old_unique_variable = 123
    return cv_wrappers.CLAHE_Enhancement(img)
""".strip()
        previous_process = """
def process(img, trial, cache):
    return cv_wrappers.Bilateral_Filter(img)
""".strip()
        messages = [
            {"role": "user", "content": "提升暗部细节"},
            {
                "role": "assistant",
                "content": "round 1 done",
                "image": object(),
                "eval_code": "def evaluate(img):\n    return 1.0",
                "process_code": older_process,
                "best_params": {"clip": 2.0},
            },
            {"role": "user", "content": "噪声太明显"},
            {
                "role": "assistant",
                "content": "round 2 done",
                "image": object(),
                "eval_code": "def evaluate(img):\n    return 2.0",
                "process_code": previous_process,
                "best_params": {"d": 5},
            },
            {"role": "user", "content": "还是太糊"},
        ]

        with patch.object(components.st, "session_state", SimpleNamespace(messages=messages)):
            prompt = components.generate_user_prompt(
                "还是太糊",
                include_process=True,
                include_evaluate=True,
                step_by_step=False,
            )

        self.assertIn("--- 初始用户要求 ---", prompt)
        self.assertIn("提升暗部细节", prompt)
        self.assertIn("--- 历史尝试摘要", prompt)
        self.assertIn("CLAHE_Enhancement", prompt)
        self.assertIn("噪声太明显", prompt)
        self.assertNotIn("old_unique_variable", prompt)
        self.assertIn("--- 上一轮使用的图像处理代码 ---", prompt)
        self.assertIn("Bilateral_Filter", prompt)
        self.assertIn("--- 本轮用户最新反馈/要求 ---", prompt)
        self.assertIn("还是太糊", prompt)


if __name__ == "__main__":
    unittest.main()
