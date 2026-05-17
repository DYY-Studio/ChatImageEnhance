import unittest

from core.orchestrator import Orchestrator


class ToolMakerRuntimeContextTests(unittest.TestCase):
    def test_operator_preference_is_removed_from_toolmaker_context(self):
        context = """
--- 运行时约束 ---
深度学习处理: enabled
处理算子偏好: 偏好深度学习
处理算子偏好：仅传统
处理设备偏好: cuda
性能档位偏好: balanced
---
""".strip()

        sanitized = Orchestrator._sanitize_toolmaker_runtime_context(context)

        self.assertNotIn("处理算子偏好", sanitized)
        self.assertIn("深度学习处理: enabled", sanitized)
        self.assertIn("处理设备偏好: cuda", sanitized)
        self.assertIn("性能档位偏好: balanced", sanitized)


if __name__ == "__main__":
    unittest.main()
