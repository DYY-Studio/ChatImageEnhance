import streamlit as st
import optuna
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_metric

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

def create_image_comparison_widget(original_img: np.ndarray, enhanced_img: np.ndarray, 
                                   original_caption: str = "原图", enhanced_caption: str = "增强结果",
                                   unique_key_prefix: str = None):
    """
    创建多模式图像对比组件（简化版）
    
    参数:
        original_img: 原始图像 (BGR格式)
        enhanced_img: 增强后的图像 (BGR格式)
        original_caption: 原始图像的标题
        enhanced_caption: 增强图像的标题
        unique_key_prefix: 唯一键前缀，用于避免多个组件之间的key冲突
    """
    # 生成唯一key前缀
    if unique_key_prefix is None:
        unique_key_prefix = f"{id(original_img)}_{id(enhanced_img)}"
    
    # 对比模式选择器 - 仅保留并排对比和标签页切换
    comparison_mode = st.radio(
        "🔍 选择对比模式",
        options=["📊 并排对比", "📑 标签页切换"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"comparison_mode_{unique_key_prefix}"
    )
    
    if comparison_mode == "📊 并排对比":
        _render_side_by_side(original_img, enhanced_img, original_caption, enhanced_caption, unique_key_prefix)
    elif comparison_mode == "📑 标签页切换":
        _render_tab_comparison(original_img, enhanced_img, original_caption, enhanced_caption, unique_key_prefix)


def _render_side_by_side(original_img: np.ndarray, enhanced_img: np.ndarray,
                         original_caption: str, enhanced_caption: str, unique_key_prefix: str = None):
    """并排对比模式 - 增强版"""
    if unique_key_prefix is None:
        unique_key_prefix = f"{id(original_img)}_{id(enhanced_img)}"
    
    st.markdown("### 📊 并排对比")
    st.caption("左右并列显示，直观对比整体效果差异")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📷 {original_caption}**")
        st.image(original_img, caption=None, channels="BGR", use_container_width=True)
    
    with col2:
        st.markdown(f"**✨ {enhanced_caption}**")
        st.image(enhanced_img, caption=None, channels="BGR", use_container_width=True)
    
    # 添加差异统计信息（默认折叠）
    with st.expander("📈 查看差异统计", expanded=False, key=f"diff_stats_{unique_key_prefix}"):
        _show_difference_stats(original_img, enhanced_img)


def _render_tab_comparison(original_img: np.ndarray, enhanced_img: np.ndarray,
                           original_caption: str, enhanced_caption: str, unique_key_prefix: str = None):
    """标签页切换对比模式 - 增强版"""
    if unique_key_prefix is None:
        unique_key_prefix = f"{id(original_img)}_{id(enhanced_img)}"
    
    st.markdown("### 📑 标签页切换")
    st.caption("通过标签页分别查看两张图片，避免视觉干扰")
    
    tab1, tab2 = st.tabs([f"📷 {original_caption}", f"✨ {enhanced_caption}"])
    
    with tab1:
        st.image(original_img, caption=original_caption, channels="BGR", use_container_width=True)
        st.caption("这是原始未处理的图像")
    
    with tab2:
        st.image(enhanced_img, caption=enhanced_caption, channels="BGR", use_container_width=True)
        st.caption("这是经过优化处理后的图像")


def _show_difference_stats(original_img: np.ndarray, enhanced_img: np.ndarray):
    """显示两张图片的差异统计"""
    mse = np.mean((original_img - enhanced_img) ** 2)
    psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # 使用 skimage 计算 SSIM
    try:
        ssim = ssim_metric(original_img, enhanced_img, channel_axis=-1, data_range=255)
    except Exception as e:
        # 如果失败，尝试灰度图模式
        if len(original_img.shape) == 3:
            original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            ssim = ssim_metric(original_gray, enhanced_gray, data_range=255)
        else:
            ssim = 0.0
    
    st.markdown(f"""```
MSE:  {mse:.4f}
PSNR: {psnr:.2f} dB
SSIM: {ssim:.4f}
```""")

class StOptunaCallbackImg:
    def __init__(self, 
        n_trials: int, 
        progress_bar: DeltaGenerator, 
        status_text: DeltaGenerator, 
        table_placeholder: DeltaGenerator,
        image: DeltaGenerator,
        best_queue: Queue[np.ndarray],
        previous_best_bgr: np.ndarray = None
    ):
        self.n_trials = n_trials
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.table_placeholder = table_placeholder
        self.image = image
        self.best_queue = best_queue
        self.previous_best_bgr = previous_best_bgr

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
                    create_image_comparison_widget(
                        self.previous_best_bgr, 
                        best_img_bgr,
                        "上一轮结果",
                        "当前最佳",
                        unique_key_prefix=f"optuna_callback_{id(self.previous_best_bgr)}_{id(best_img_bgr)}"
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
