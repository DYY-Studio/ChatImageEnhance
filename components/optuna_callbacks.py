import streamlit as st
import optuna
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_metric

from streamlit.delta_generator import DeltaGenerator
from queue import Queue

from utils import get_thumbnail_img, get_thumbnail_img_nocache

import plotly.graph_objects as go
import plotly.express as px
from optuna.visualization import plot_param_importances, plot_optimization_history
from optuna.visualization.matplotlib import plot_param_importances as plt_plot_param_importances
from optuna.visualization.matplotlib import plot_optimization_history as plt_plot_optimization_history
import matplotlib.pyplot as plt

# 设置matplotlib中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class StOptunaCallback:
    def __init__(self, 
        n_trials: int, 
        progress_bar: DeltaGenerator, 
        status_text: DeltaGenerator, 
        table_placeholder: DeltaGenerator,
        # 新增可视化占位符
        importance_placeholder: DeltaGenerator = None,
        history_placeholder: DeltaGenerator = None
    ):
        self.n_trials = n_trials
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.table_placeholder = table_placeholder
        # 新增属性
        self.importance_placeholder = importance_placeholder
        self.history_placeholder = history_placeholder
       
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

        # 新增：更新超参数重要性图和优化历史图
        self._update_visualizations(study)
    
    def _update_visualizations(self, study: optuna.study.Study):
        """更新超参数重要性和优化历史可视化"""
        # 至少有2个trial才绘制图表
        if len(study.trials) < 2:
            return
        
        # 1. 绘制超参数重要性图
        if self.importance_placeholder is not None:
            try:
                with self.importance_placeholder.container():
                    st.subheader("超参数重要性")
                    # 使用plotly绘制交互式图表
                    fig_importance = plot_param_importances(study)
                    # 调整布局
                    fig_importance.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(size=10)
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            except Exception as e:
                st.warning(f"超参数重要性图绘制失败: {str(e)}")
        
        # 2. 绘制优化历史图
        if self.history_placeholder is not None:
            try:
                with self.history_placeholder.container():
                    st.subheader("优化历史")
                    # 使用plotly绘制交互式图表
                    fig_history = plot_optimization_history(study)
                    # 调整布局
                    fig_history.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(size=10)
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
            except Exception as e:
                st.warning(f"优化历史图绘制失败: {str(e)}")

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
        interpolate: int = cv2.INTER_AREA,
         # 新增：接收可视化占位符
        importance_placeholder: DeltaGenerator = None,
        history_placeholder: DeltaGenerator = None
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
    
    # 新增：可视化图表占位符
        self.importance_placeholder = importance_placeholder
        self.history_placeholder = history_placeholder

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
                        get_thumbnail_img(self.previous_best_bgr, 'binary', self.max_side, self.interpolate), 
                        get_thumbnail_img_nocache(best_img_bgr, 'binary', self.max_side, self.interpolate),
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

        # 新增：更新可视化图表
        self._update_visualizations(study)

            # === 新增：最后一次试验时生成图表 JSON 并放入队列 ===
        # 使用 _charts_sent 标志位防止重复发送（只在最后一次试验时执行一次）
        if not hasattr(self, '_charts_sent') and len(study.trials) >= self.n_trials:
            self._charts_sent = True   # 标记已发送
            importance_json = None
            history_json = None

            # 生成超参数重要性图的 JSON
            if self.importance_placeholder is not None:
                try:
                    fig_imp = plot_param_importances(study)
                    fig_imp.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(size=10)
                    )   
                    importance_json = fig_imp.to_json()
                except Exception as e:
                    st.warning(f"超参数重要性图 JSON 生成失败: {str(e)}")

            # 生成优化历史图的 JSON
            if self.history_placeholder is not None:
                try:
                    fig_hist = plot_optimization_history(study)
                    fig_hist.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(size=10)
                    )
                    history_json = fig_hist.to_json()
                except Exception as e:
                    st.warning(f"优化历史图 JSON 生成失败: {str(e)}")

            # 将图表 JSON 放入队列（用一个特殊元组标记，与图像区分）
            self.best_queue.put(('CHARTS', importance_json, history_json))
        # === 新增结束 ===

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
            st.image(original_img_bytes, caption=original_caption, width='stretch')
        with comp_tab3:
            st.image(enhanced_img_bytes, caption=enhanced_caption, width='stretch')


    def _render_side_by_side(self, 
        original_img_bytes: bytes, enhanced_img_bytes: bytes,
        original_caption: str = "原图", enhanced_caption: str = "增强结果"
    ):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_img_bytes, caption=original_caption, width='stretch')
        
        with col2:
            st.image(enhanced_img_bytes, caption=enhanced_caption, width='stretch')

    def _update_visualizations(self, study: optuna.study.Study):
        """更新超参数重要性和优化历史可视化（与StOptunaCallback逻辑一致）"""
        # 至少有2个trial才绘制图表
        if len(study.trials) < 2:
            return
        
        # 1. 绘制超参数重要性图
        if self.importance_placeholder is not None:
            try:
                with self.importance_placeholder.container():
                    st.subheader("超参数重要性")
                    # 使用plotly绘制交互式图表
                    fig_importance = plot_param_importances(study)
                    # 调整布局
                    fig_importance.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(size=10)
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            except Exception as e:
                st.warning(f"超参数重要性图绘制失败: {str(e)}")
        
        # 2. 绘制优化历史图
        if self.history_placeholder is not None:
            try:
                with self.history_placeholder.container():
                    st.subheader("优化历史")
                    # 使用plotly绘制交互式图表
                    fig_history = plot_optimization_history(study)
                    # 调整布局
                    fig_history.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=10, b=20),
                        font=dict(size=10)
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
            except Exception as e:
                st.warning(f"优化历史图绘制失败: {str(e)}")