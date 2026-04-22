import streamlit as st
import optuna
import numpy as np

from streamlit.delta_generator import DeltaGenerator
from queue import Queue

class StOptunaCallback:
    def __init__(self, 
        n_trials: int, 
        progress_bar: DeltaGenerator, 
        status_text: DeltaGenerator, 
        table_placeholder: DeltaGenerator,
    ):
        self.n_trials = n_trials
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.table_placeholder = table_placeholder

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # 计算进度
        current_trial = len(study.trials)
        progress = current_trial / self.n_trials
        
        # 更新进度条
        self.progress_bar.progress(progress)
        try:
            self.status_text.text(f"正在进行第 {current_trial}/{self.n_trials} 轮优化... "
                                f"当前最佳值: {study.best_value:.4f}")
        except:
            self.status_text.text(f"正在进行第 {current_trial}/{self.n_trials} 轮优化... ")
        
        # 更新实时数据表格 (展示最近的 10 条记录)
        df = study.trials_dataframe().tail(10)
        if not df.empty:
            cols_to_show = df.filter(regex='^(value|state|params_.*)$').columns.tolist()
            display_df = df[cols_to_show].copy()
            
            display_df.columns = [c.replace('params_', '') for c in display_df.columns]
            self.table_placeholder.dataframe(display_df)

class StOptunaCallbackImg:
    def __init__(self, 
        n_trials: int, 
        progress_bar: DeltaGenerator, 
        status_text: DeltaGenerator, 
        table_placeholder: DeltaGenerator,
        image: DeltaGenerator,
        best_queue: Queue[np.ndarray]
    ):
        self.n_trials = n_trials
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.table_placeholder = table_placeholder
        self.image = image
        self.best_queue = best_queue

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # 计算进度
        current_trial = len(study.trials)
        progress = current_trial / self.n_trials
        
        # 更新进度条
        self.progress_bar.progress(progress)
        try:
            self.status_text.text(f"正在进行第 {current_trial}/{self.n_trials} 轮优化... "
                                f"当前最佳值: {study.best_value:.4f}")
        except:
            self.status_text.text(f"正在进行第 {current_trial}/{self.n_trials} 轮优化... ")

        # 更新最佳图像（如果有）
        best_img_bgr = None
        while not self.best_queue.empty():
            best_img_bgr = self.best_queue.get()
        if best_img_bgr is not None:
            self.image.image(best_img_bgr, channels="BGR")
        
        # 更新实时数据表格 (展示最近的 10 条记录)
        df = study.trials_dataframe().tail(10)
        if not df.empty:
            cols_to_show = df.filter(regex='^(value|state|params_.*)$').columns.tolist()
            display_df = df[cols_to_show].copy()
            
            display_df.columns = [c.replace('params_', '') for c in display_df.columns]
            self.table_placeholder.dataframe(display_df)