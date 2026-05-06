import optuna
import numpy as np
import logging

from sandbox.executor import SandboxExecutor
from typing import Iterable, Callable
from queue import Queue

logger = logging.getLogger("BayesianOptimizer")

class BayesianOptimizer:
    """
    在 LLM 确定代码结构后，接管底层参数寻优。
    """
    def __init__(self, executor: SandboxExecutor):
        self.executor = executor
        self.study = optuna.create_study(direction="maximize")

    def run_inner_loop(self, 
        code_str: str, 
        evaluate_code_str: str,
        base_img: np.ndarray, 
        orig_img: np.ndarray,
        n_trials: int = 30, 
        callbacks: Iterable[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None = None,
    ) -> dict:
        """
        针对固定的代码拓扑，运行 Optuna 寻找最优参数。

        返回: {'best_score': float, 'best_params': dict, 'best_img': np.ndarray, 'n_trials_used': int}
        """
        def objective(trial):
            try:
                result_img = self.executor.execute_pipeline(code_str, base_img, trial)
                score = self.executor.execute_evaluate(evaluate_code_str, result_img, base_img)
                
                if score <= -5000.0: 
                    raise optuna.TrialPruned()
                return score
            except optuna.TrialPruned:
                pass
            except Exception as e:
                logger.error("CODE EXEC ERROR", e)
                raise optuna.TrialPruned() # 代码执行错误，修剪该 trial

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
        
        # ===== [新增] 记录实际使用的trial数 =====
        n_trials_used = len(study.trials)
        
        try:
            return {
                "best_score": study.best_value,
                "best_params": study.best_params,
                "best_img": self.executor.execute_pipeline_direct(
                    code_str, orig_img, study.best_params
                ),
                # ===== [新增] 返回实际使用的trial数 =====
                "n_trials_used": n_trials_used
            }
        except:
            return {
                "best_score": None,
                "best_params": None,
                "best_img": None,
                # ===== [新增] 即使出错也返回实际trial数 =====
                "n_trials_used": n_trials_used
            }

    def run_inner_loop_stream(self, 
        code_str: str, 
        evaluate_code_str: str,
        base_img: np.ndarray, 
        orig_img: np.ndarray,
        best_queue: Queue,
        n_trials: int = 30, 
        callbacks: Iterable[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] | None = None,
    ) -> dict:
        """
        针对固定的代码拓扑，运行 Optuna 寻找最优参数。

        返回: {'best_score': float, 'best_params': dict, 'best_img': np.ndarray, 'n_trials_used': int}
        """

        logger.info(f"BASE: {base_img.shape[1]}x{base_img.shape[0]}")
        if orig_img is not None:
            logger.info(f"ORIG: {orig_img.shape[1]}x{orig_img.shape[0]}")

        def objective(trial: optuna.trial.Trial):
            nonlocal code_str, evaluate_code_str, base_img
            try:
                result_img = self.executor.execute_pipeline(code_str, base_img, trial)
                score = self.executor.execute_evaluate(evaluate_code_str, result_img, base_img)
                
                if score <= -5000.0: 
                    raise optuna.TrialPruned()
                study = trial.study
                try:
                    if score > study.best_value:
                        best_queue.put(result_img)
                except:
                    best_queue.put(result_img)
                return score
            except optuna.TrialPruned:
                pass
            except Exception as e:
                logger.error("CODE EXEC ERROR", e)
                raise optuna.TrialPruned()  # 代码执行错误，修剪该 trial

        self.study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
        
        # ===== [新增] 记录实际使用的trial数并打印日志 =====
        n_trials_used = len(self.study.trials)
        logger.info(f"实际使用 trial 数: {n_trials_used} / {n_trials}")
        
        try:
            return {
                "best_score": self.study.best_value,
                "best_params": self.study.best_params,
                "best_img": self.executor.execute_pipeline_direct(
                    code_str, orig_img, self.study.best_params
                ),
                # ===== [新增] 返回实际使用的trial数 =====
                "n_trials_used": n_trials_used
            }
        except:
            return {
                "best_score": None,
                "best_params": None,
                "best_img": None,
                # ===== [新增] 即使出错也返回实际trial数 =====
                "n_trials_used": n_trials_used
            }