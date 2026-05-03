import streamlit as st
import optuna
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_metric

from streamlit.delta_generator import DeltaGenerator
from queue import Queue

from utils import get_thumbnail_img, get_thumbnail_img_nocache

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
        best_queue: Queue[np.ndarray],
        previous_best_bgr: np.ndarray = None,
        compare_to_raw: bool = True,
        max_side: int = 800,
        interpolate: int = cv2.INTER_AREA
    ):
        self.n_trials = n_trials
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.table_placeholder = table_placeholder
        self.image = image
        self.best_queue = best_queue

        self.previous_best_bgr = previous_best_bgr
        self.compare_to_raw = compare_to_raw

        self.max_side = max_side
        self.interpolate = interpolate

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
            # 如果有上一轮的图片，则使用增强的对比组件
            if self.previous_best_bgr is not None:
                with self.image.container():
                    self.create_image_comparison_widget(
                        get_thumbnail_img(self.previous_best_bgr, self.max_side, self.interpolate), 
                        get_thumbnail_img_nocache(best_img_bgr, self.max_side, self.interpolate),
                        "上一轮结果" if not self.compare_to_raw else "原图",
                        "当前最佳",
                    )
            else:
                self.image.image(best_img_bgr, channels="BGR")
        
        # 更新实时数据表格 (展示最近的 10 条记录)
        df = study.trials_dataframe().tail(10)
        if not df.empty:
            cols_to_show = df.filter(regex='^(value|state|params_.*)$').columns.tolist()
            display_df = df[cols_to_show].copy()
            
            display_df.columns = [c.replace('params_', '') for c in display_df.columns]
            self.table_placeholder.dataframe(display_df)

    def create_image_comparison_widget(self,
        original_img_bytes: bytes, enhanced_img_bytes: bytes, 
        original_caption: str = "原图", enhanced_caption: str = "增强结果",
    ):
        """
        创建多模式图像对比组件（简化版）
        
        参数:
            original_img: 原始图像 (BGR格式)
            enhanced_img: 增强后的图像 (BGR格式)
            original_caption: 原始图像的标题
            enhanced_caption: 增强图像的标题
        """

        comp_tab1, comp_tab2, comp_tab3 = st.tabs(
            ["并排对比", original_caption, enhanced_caption],
        )
        
        with comp_tab1:
            self._render_side_by_side(original_img_bytes, enhanced_img_bytes, original_caption, enhanced_caption)
        with comp_tab2:
            st.image(original_img_bytes, caption=original_caption, use_container_width=True)
        with comp_tab3:
            st.image(enhanced_img_bytes, caption=enhanced_caption, use_container_width=True)


    def _render_side_by_side(self, 
        original_img_bytes: bytes, enhanced_img_bytes: bytes,
        original_caption: str = "原图", enhanced_caption: str = "增强结果"
    ):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_img_bytes, caption=original_caption, use_container_width=True)
        
        with col2:
            st.image(enhanced_img_bytes, caption=enhanced_caption, use_container_width=True)

