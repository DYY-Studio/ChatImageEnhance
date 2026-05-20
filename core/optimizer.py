import optuna
import numpy as np
import gc
import logging
import re
import traceback

from sandbox.executor import SandboxExecutor
from sandbox.runtime_dependencies import RuntimeDependencyManager
from typing import Iterable, Callable
from collections import deque
from optuna.trial import TrialState

logger = logging.getLogger("BayesianOptimizer")

class BayesianOptimizer:
    """
    在 LLM 确定代码结构后，接管底层参数寻优。
    """
    def __init__(self, executor: SandboxExecutor):
        self.executor = executor
        self.study = optuna.create_study(direction="maximize")

    @staticmethod
    def _cleanup_unicache(unicache: dict):
        """释放 unicache 中残留的深度学习模型和 Tensor，防止 GPU 显存泄漏。"""
        if not unicache:
            return
        # 跳过运行时元数据 key
        runtime_keys = {k for k in unicache if k.startswith('__runtime')}
        for key in list(unicache.keys()):
            if key in runtime_keys:
                continue
            try:
                del unicache[key]
            except Exception:
                pass
        unicache.clear()
        gc.collect()
        RuntimeDependencyManager.release_device_memory()

    def _has_trial(self, code_str: str):
        return re.search(r"\w+?\.suggest_(int|float|categorical|(?:discrete_|log)?uniform)", code_str) is not None

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
        unicache = dict()
        last_error: str | None = None

        def objective(trial):
            nonlocal unicache, last_error
            try:
                result_img = self.executor.execute_pipeline(code_str, base_img, trial, unicache)
                score = self.executor.execute_evaluate(evaluate_code_str, result_img, base_img)
                
                if score <= -5000.0: 
                    raise optuna.TrialPruned()
                return score
            except optuna.TrialPruned:
                raise
            except Exception as e:
                last_error = traceback.format_exc()
                logger.error("CODE EXEC ERROR: %s", e)
                raise optuna.TrialPruned() # 代码执行错误，修剪该 trial

        if not self._has_trial(code_str):
            try:
                result = {
                    "best_score": None,
                    "best_params": None,
                    "best_img": self.executor.execute_pipeline_direct(
                        code_str, orig_img, {}, unicache
                    ),
                    "n_trials_used": 0
                }
                self._cleanup_unicache(unicache)
                return result
            except Exception:
                self._cleanup_unicache(unicache)
                return {
                    "best_score": None,
                    "best_params": None,
                    "best_img": None,
                    "n_trials_used": 0,
                    "error": traceback.format_exc(),
                }

        study = self.study
        study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
        
        # ===== [新增] 记录实际使用的trial数 =====
        n_trials_used = len(study.trials)
        completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if not completed_trials:
            self._cleanup_unicache(unicache)
            return {
                "best_score": None,
                "best_params": None,
                "best_img": None,
                "n_trials_used": n_trials_used,
                "error": last_error or "所有 Optuna trial 都失败或被剪枝，未得到可用图像。",
            }
        
        try:
            result = {
                "best_score": study.best_value,
                "best_params": study.best_params,
                "best_img": self.executor.execute_pipeline_direct(
                    code_str, orig_img, study.best_params, unicache
                ),
                # ===== [新增] 返回实际使用的trial数 =====
                "n_trials_used": n_trials_used
            }
            self._cleanup_unicache(unicache)
            return result
        except:
            last_error = traceback.format_exc()
            self._cleanup_unicache(unicache)
            return {
                "best_score": None,
                "best_params": None,
                "best_img": None,
                # ===== [新增] 即使出错也返回实际trial数 =====
                "n_trials_used": n_trials_used,
                "error": last_error,
            }

    def run_inner_loop_stream(self, 
        code_str: str, 
        evaluate_code_str: str,
        base_img: np.ndarray, 
        orig_img: np.ndarray,
        best_queue: deque[np.ndarray],
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

        unicache = dict()
        last_error: str | None = None

        def objective(trial: optuna.trial.Trial):
            nonlocal code_str, evaluate_code_str, base_img, unicache, last_error
            try:
                result_img = self.executor.execute_pipeline(code_str, base_img, trial, unicache)
                score = self.executor.execute_evaluate(evaluate_code_str, result_img, base_img)
                
                if score <= -5000.0: 
                    raise optuna.TrialPruned()
                study = trial.study
                try:
                    if score > study.best_value:
                        best_queue.append(result_img)
                except:
                    best_queue.append(result_img)
                return score
            except optuna.TrialPruned:
                raise
            except Exception as e:
                last_error = traceback.format_exc()
                logger.error("CODE EXEC ERROR: %s", e)
                raise optuna.TrialPruned()  # 代码执行错误，修剪该 trial
            
        # 如果代码中没有 trial 可搜索的内容，则只执行一遍就返回
        if not self._has_trial(code_str):
            try:
                result = {
                    "best_score": None,
                    "best_params": None,
                    "best_img": self.executor.execute_pipeline_direct(
                        code_str, orig_img, {}, unicache
                    ),
                    "n_trials_used": 0
                }
                self._cleanup_unicache(unicache)
                return result
            except Exception:
                self._cleanup_unicache(unicache)
                return {
                    "best_score": None,
                    "best_params": None,
                    "best_img": None,
                    "n_trials_used": 0,
                    "error": traceback.format_exc(),
                }

        study = self.study
        study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
        
        # ===== [新增] 记录实际使用的trial数并打印日志 =====
        n_trials_used = len(study.trials)
        logger.info(f"实际使用 trial 数: {n_trials_used} / {n_trials}")
        completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        if not completed_trials:
            self._cleanup_unicache(unicache)
            return {
                "best_score": None,
                "best_params": None,
                "best_img": None,
                "n_trials_used": n_trials_used,
                "error": last_error or "所有 Optuna trial 都失败或被剪枝，未得到可用图像。",
            }
        
        try:
            result = {
                "best_score": study.best_value,
                "best_params": study.best_params,
                "best_img": self.executor.execute_pipeline_direct(
                    code_str, orig_img, study.best_params, unicache
                ),
                # ===== [新增] 返回实际使用的trial数 =====
                "n_trials_used": n_trials_used
            }
            self._cleanup_unicache(unicache)
            return result
        except:
            last_error = traceback.format_exc()
            self._cleanup_unicache(unicache)
            return {
                "best_score": None,
                "best_params": None,
                "best_img": None,
                # ===== [新增] 即使出错也返回实际trial数 =====
                "n_trials_used": n_trials_used,
                "error": last_error,
            }
