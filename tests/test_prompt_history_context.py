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

    def test_generate_user_prompt_includes_context_for_previous_result_input(self):
        messages = [
            {"role": "user", "content": "让照片更清晰"},
            {
                "role": "assistant",
                "content": "round 1 done",
                "image": object(),
                "eval_code": "def evaluate(img):\n    return 1.0",
                "process_code": "def process(img, trial, cache):\n    return cv_wrappers.Unsharp_Mask(img)",
                "best_params": {"amount": 1.2},
                "input_from_previous": False,
                "input_source": "original",
            },
            {"role": "user", "content": "在这个基础上再柔和一点"},
        ]

        with patch.object(components.st, "session_state", SimpleNamespace(messages=messages)):
            prompt = components.generate_user_prompt(
                "在这个基础上再柔和一点",
                include_process=True,
                include_evaluate=False,
                step_by_step=True,
            )

        self.assertIn("--- 本轮输入图像来源 ---", prompt)
        self.assertIn("上一轮结果图像", prompt)
        self.assertIn("--- 上一轮使用的图像处理代码 ---", prompt)
        self.assertIn("Unsharp_Mask", prompt)

    def test_export_script_chains_previous_result_rounds(self):
        messages = [
            {"role": "user", "content": "初始"},
            {
                "role": "assistant",
                "content": "round 1",
                "image": object(),
                "process_code": """
def process(img, trial, cache):
    amount = trial.suggest_int("a", 1, 3)
    return img + amount
""",
                "best_params": {"a": 1},
                "input_from_previous": False,
                "input_source": "original",
            },
            {"role": "user", "content": "继续"},
            {
                "role": "assistant",
                "content": "round 2",
                "image": object(),
                "process_code": """
def process(img, trial, cache):
    amount = trial.suggest_int("b", 1, 3)
    return img + amount
""",
                "best_params": {"b": 2},
                "input_from_previous": True,
                "input_source": "previous_result",
            },
            {"role": "user", "content": "再继续"},
            {
                "role": "assistant",
                "content": "round 3",
                "image": object(),
                "process_code": """
def process(img, trial, cache):
    amount = trial.suggest_int("c", 1, 3)
    return img + amount
""",
                "best_params": {"c": 3},
                "input_from_previous": True,
                "input_source": "previous_result",
            },
        ]

        script = components.build_export_script_for_message(messages, 5)

        self.assertIsNotNone(script)
        self.assertIn("def process_step_1(img, params, cache):", script)
        self.assertIn("def process_step_2(img, params, cache):", script)
        self.assertIn("def process_step_3(img, params, cache):", script)
        self.assertIn('params["a"]', script)
        self.assertIn('params["b"]', script)
        self.assertIn('params["c"]', script)
        self.assertIn("best_params_sequence = [{'a': 1}, {'b': 2}, {'c': 3}]", script)
        self.assertLess(script.index("process_step_1"), script.index("process_step_2"))
        self.assertLess(script.index("process_step_2"), script.index("process_step_3"))
        self.assertIn("out = process_step_1(out, params_sequence[0], cache)", script)
        self.assertIn("out = process_step_2(out, params_sequence[1], cache)", script)
        self.assertIn("out = process_step_3(out, params_sequence[2], cache)", script)

    def test_export_script_uses_only_current_round_for_original_input(self):
        messages = [
            {
                "role": "assistant",
                "content": "round 1",
                "image": object(),
                "process_code": "def process(img, trial, cache):\n    return img",
                "best_params": {},
                "input_from_previous": False,
                "input_source": "original",
            },
            {
                "role": "assistant",
                "content": "round 2",
                "image": object(),
                "process_code": """
def process(img, trial, cache):
    amount = trial.suggest_int("b", 1, 3)
    return img + amount
""",
                "best_params": {"b": 2},
                "input_from_previous": False,
                "input_source": "original",
            },
        ]

        script = components.build_export_script_for_message(messages, 1)

        self.assertIsNotNone(script)
        self.assertIn("def process_step_1(img, params, cache):", script)
        self.assertNotIn("def process_step_2", script)
        self.assertIn("best_params_sequence = [{'b': 2}]", script)


if __name__ == "__main__":
    unittest.main()
