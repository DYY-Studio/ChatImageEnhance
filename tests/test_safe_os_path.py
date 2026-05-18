import unittest

import numpy as np

from sandbox.code_checker import SecurityViolation
from sandbox.executor import SandboxExecutor


class SandboxSafeOSPathTests(unittest.TestCase):
    def test_sandbox_exposes_safe_os_path_proxy(self):
        code = """
def process(img: np.ndarray, trial: optuna.Trial, cache: dict) -> np.ndarray:
    joined = os.path.join('models', 'weights.pth')
    normalized = os_path.normpath('models/../weights/file.pth')
    cache['safe_path_values'] = (
        joined,
        os.path.basename(normalized),
        os_path.splitext(normalized)[1],
        os.path.sep,
    )
    return img
"""
        executor = SandboxExecutor()
        img = np.zeros((4, 4), dtype=np.uint8)
        cache = {}

        result = executor.execute_pipeline_direct(code, img, {}, cache)

        self.assertTrue(np.array_equal(result, img))
        joined, basename, ext, sep = cache["safe_path_values"]
        self.assertTrue(joined.endswith(f"models{sep}weights.pth"))
        self.assertEqual(basename, "file.pth")
        self.assertEqual(ext, ".pth")

    def test_sandbox_supports_safe_os_path_import_forms(self):
        code = """
def process(img: np.ndarray, trial: optuna.Trial, cache: dict) -> np.ndarray:
    import os
    import os.path as osp
    from os import path
    from os.path import basename, join

    cache['imported_path_values'] = (
        basename(join('models', 'weights.pth')),
        path.splitext('image.png')[1],
        osp.dirname('models/weights.pth'),
        os.path.normpath('a/../b'),
    )
    return img
"""
        executor = SandboxExecutor()
        img = np.zeros((4, 4), dtype=np.uint8)
        cache = {}

        executor.execute_pipeline_direct(code, img, {}, cache)

        basename, ext, dirname, normalized = cache["imported_path_values"]
        self.assertEqual(basename, "weights.pth")
        self.assertEqual(ext, ".png")
        self.assertEqual(dirname, "models")
        self.assertEqual(normalized, "b")

    def test_sandbox_blocks_filesystem_probe_through_os_path(self):
        code = """
def process(img: np.ndarray, trial: optuna.Trial, cache: dict) -> np.ndarray:
    cache['exists'] = os.path.exists('models/weights.pth')
    return img
"""
        executor = SandboxExecutor()

        with self.assertRaises(SecurityViolation):
            executor.prepare_code(code)


if __name__ == "__main__":
    unittest.main()
