import unittest

import numpy as np
from optuna.trial import TrialState

from core.optimizer import BayesianOptimizer


class FakeExecutor:
    def execute_pipeline(self, code_str, img, trial, cache):
        trial.suggest_float("x", 0.0, 1.0)
        return img

    def execute_evaluate(self, code_str, img, evaluator):
        return -6000.0


class BayesianOptimizerTests(unittest.TestCase):
    def test_low_score_trials_are_pruned_not_failed(self):
        optimizer = BayesianOptimizer(FakeExecutor())
        img = np.zeros((8, 8, 3), dtype=np.uint8)

        result = optimizer.run_inner_loop_stream(
            "def process(img, trial):\n    trial.suggest_float('x', 0.0, 1.0)\n    return img",
            "def evaluate(img):\n    return -6000.0",
            img,
            img,
            best_queue=[],
            n_trials=2,
        )

        self.assertIsNone(result["best_img"])
        self.assertEqual(result["n_trials_used"], 2)
        self.assertTrue(optimizer.study.trials)
        self.assertTrue(all(trial.state == TrialState.PRUNED for trial in optimizer.study.trials))


if __name__ == "__main__":
    unittest.main()
