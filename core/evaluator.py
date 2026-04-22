import cv2
import json
import yaml
import numpy as np
from skimage.metrics import structural_similarity as ssim

class Evaluator:
    """
    基于弱参考的图像质量评估器。
    结合了清晰度（驱动）、保真度（约束）和自然度/平滑度（惩罚）。
    """
    def __init__(self, original_img: np.ndarray):
        self.original_img = original_img
        # 预先将原图转为灰度图，节省后续计算 SSIM 的开销
        self.gray_original = self._to_gray(original_img)
        
        # 记录原图的基准指标，可用于计算相对提升比例
        self.base_sharpness = self.compute_sharpness(self.gray_original)
        self.base_tv = self.compute_tv(self.gray_original)
        self.base_entropy = self.compute_entropy(self.gray_original)
        self.base_clipping = self.compute_clipping(self.gray_original)
        self.base_saturation = self.compute_saturation_metrics(self.original_img)
        self.base_color_cast = self.compute_color_cast(self.original_img)
        self.base_snr = self.compute_snr(self.gray_original)
        self.base_hf_ratio = self.compute_high_freq_ratio(self.gray_original)
        
        self.brisque_obj = cv2.quality.QualityBRISQUE.create(
            "brisque_model_live.yml", 
            "brisque_range_live.yml"
        ) if hasattr(cv2, "quality") else None

        self.base_brisque = self.compute_brisque(self.original_img)
        self.base_brightness = float(np.mean(self.gray_original))

        self.profile_dict = self._generate_profile()

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """安全地将图像转换为灰度图。"""
        if len(img.shape) == 3:
            # 假设输入为 BGR 格式 (OpenCV 默认)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _align_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """防御性编程：确保评估时图像尺寸一致，防止 Agent 改变了输出尺寸导致报错。"""
        if img1.shape != img2.shape:
            return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img2

    def compute_sharpness(self, img: np.ndarray) -> float:
        """
        计算清晰度：拉普拉斯方差 (Variance of Laplacian)。
        值越大，代表图像高频边缘信息越丰富。
        """
        gray = self._to_gray(img)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def compute_fidelity(self, img: np.ndarray) -> float:
        """
        计算保真度：结构相似性 (SSIM)。
        返回 0 到 1 之间的值，越接近 1 代表与原图结构越相似。
        """
        gray = self._to_gray(img)
        gray = self._align_images(self.gray_original, gray)
        
        # data_range=255 适用于 8-bit 图像
        score, _ = ssim(self.gray_original, gray, full=True, data_range=255)
        return score

    def compute_tv(self, img: np.ndarray) -> float:
        """
        计算全变分 (Total Variation)，用于量化噪点和过度锐化的粗糙感。
        进行了尺寸归一化，确保不同分辨率的图像处于同一量级。
        """
        # 使用灰度图或多通道图均可，这里统一使用灰度计算以保持量级一致
        gray = self._to_gray(img).astype(np.float32)
        diff_x = np.abs(gray[:, :-1] - gray[:, 1:])
        diff_y = np.abs(gray[:-1, :] - gray[1:, :])
        
        # 归一化：除以像素总数
        total_pixels = gray.shape[0] * gray.shape[1]
        tv = (np.sum(diff_x) + np.sum(diff_y)) / total_pixels
        return float(tv)
    
    def compute_entropy(self, img: np.ndarray) -> float:
        """
        计算图像信息熵 (Image Entropy)。
        用于衡量图像信息的丰富度，过暗、过曝或低对比度图像熵值通常较低。
        """
        gray = self._to_gray(img)
        # 计算直方图并归一化为概率分布
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist_prob = hist / hist.sum()
        # 过滤掉概率为 0 的项，避免 log2(0) 报错
        hist_prob = hist_prob[hist_prob > 0]
        entropy = -np.sum(hist_prob * np.log2(hist_prob))
        return float(entropy)

    def compute_clipping(self, img: np.ndarray) -> float:
        """
        计算像素溢出率 (Clipping Ratio)。
        统计绝对纯黑 (0) 和纯白 (255) 像素的占比，用于严厉惩罚极端曝光调整。
        """
        gray = self._to_gray(img)
        # 统计极值像素比例
        clip_ratio = np.mean((gray == 0) | (gray == 255))
        return float(clip_ratio)

    def compute_brisque(self, img: np.ndarray) -> float:
        """
        无参考图像质量评估 (BRISQUE)。
        分数通常在 0-100 之间，分数越低代表图像感知质量越好（伪影和失真越少）。
        依赖 opencv-contrib-python。
        """
        if self.brisque_obj:
            return self.brisque_obj.compute(img)[0]
        else:
            return 50.0
        
    def compute_saturation_metrics(self, img: np.ndarray) -> float:
        """
        计算平均饱和度。
        对于黑白图像返回 0。
        """
        if len(img.shape) != 3:
            return 0.0
        # 转换到 HSV 空间，提取 S 通道 (0-255)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        return float(np.mean(s_channel))

    def compute_color_cast(self, img: np.ndarray) -> dict:
        """
        色偏检测 (Color Cast)。
        返回 R, G, B 三通道的平均值。Agent 可据此判断是否需要白平衡。
        """
        if len(img.shape) != 3:
            return {"b": 0.0, "g": 0.0, "r": 0.0}
        b, g, r = cv2.split(img)
        return {
            "b": round(float(np.mean(b)), 2),
            "g": round(float(np.mean(g)), 2),
            "r": round(float(np.mean(r)), 2)
        }

    def compute_snr(self, img: np.ndarray) -> float:
        """
        盲信噪比估算 (Blind SNR Estimation)。
        使用全局均值与标准差的简单比值计算对数信噪比（单位 dB）。
        值越大，代表信号（图像内容）越强于噪声。
        """
        gray = self._to_gray(img).astype(np.float32)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        if std_val == 0:
            return 999.0  # 避免除以 0，纯色图信噪比极高
        
        # 10 * log10((μ^2) / (σ^2))
        snr = 10 * np.log10((mean_val**2) / (std_val**2 + 1e-6))
        return float(snr)

    def compute_high_freq_ratio(self, img: np.ndarray) -> float:
        """
        高频能量占比 (High Frequency Energy Ratio)。
        通过 2D 傅里叶变换将图像转换到频域，计算高频部分能量占比。
        用于判断图像是偏平滑还是偏锐利/嘈杂。
        """
        gray = self._to_gray(img).astype(np.float32)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        # 定义低频区域半径为图像短边的 1/8
        r = min(rows, cols) // 8 
        
        # 创建高频掩膜 (中心低频区为 0，其余为 1)
        mask = np.ones((rows, cols), np.float32)
        cv2.circle(mask, (ccol, crow), r, 0, -1)
        
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy == 0:
            return 0.0
        return float(high_freq_energy / total_energy)
    
    def _generate_profile(self) -> dict:
        """
        提取图像的尺寸、亮度、对比度、清晰度和噪声估算等关键指标。
        """
        h, w = self.original_img.shape[:2]
        channels = self.original_img.shape[2] if len(self.original_img.shape) == 3 else 1
        
        gray = self.gray_original
        
        # 亮度和对比度计算
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        
        # 极暗/过曝像素比例计算 (转化为百分比)
        dark_ratio = float(np.mean(gray < 15)) * 100
        highlight_ratio = float(np.mean(gray > 240)) * 100
        
        # 组装 Profile 字典
        profile = {
            "image_profile": {
                "dimensions": {
                    "width": w, 
                    "height": h, 
                    "channels": channels
                },
                "brightness": {
                    "mean": round(mean_intensity, 2),
                    "dark_pixels_percent": round(dark_ratio, 2),
                    "highlight_pixels_percent": round(highlight_ratio, 2)
                },
                "contrast": {
                    "std_dev": round(std_intensity, 2)
                },
                "sharpness": {
                    "laplacian_variance": round(self.base_sharpness, 2)
                },
                "quality_metrics": {
                    "information_entropy": round(self.base_entropy, 2),
                    "brisque_score": round(self.base_brisque, 2),
                    "clipping_ratio": round(self.base_clipping, 4)
                },
                "color_and_saturation": {
                    "mean_saturation": round(self.base_saturation, 2),
                    "channel_means": self.base_color_cast
                },
                "frequency_and_noise": {
                    "estimated_snr_db": round(self.base_snr, 2),
                    "high_frequency_ratio": round(self.base_hf_ratio, 4),
                    "total_variation": round(self.base_tv, 2)
                }
            }
        }
        return profile

    def get_profile_json(self) -> str:
        """
        以 JSON 字符串格式输出图像特征，直接用于拼接 LLM Prompt。
        """
        return json.dumps(self.profile_dict, indent=2, ensure_ascii=False)
    
    def get_profile_yaml(self) -> str:
        return yaml.dump(self.profile_dict, indent=2, allow_unicode=True)

    def get_reward_score(self, processed_img: np.ndarray, task_weights: dict[str, float] | None = None) -> float:
        """
        综合计算 Reward 分数，供 Optuna 优化使用。
        """
        # 1. 计算三大核心指标
        processed_gray = self._to_gray(processed_img)

        if task_weights is None:
            task_weights = {
                "sharpness": 1.0, 
                "tv": -0.5, 
                "entropy": 0.0, 
                "brightness": 0.0
            }

        # 1. 计算当前处理图的指标
        cur_sharpness = self.compute_sharpness(processed_gray)
        cur_tv = self.compute_tv(processed_gray)
        cur_entropy = self.compute_entropy(processed_gray)
        cur_brightness = float(np.mean(processed_gray))
        cur_brisque = self.compute_brisque(processed_img)
        cur_clipping = self.compute_clipping(processed_gray)
        cur_saturation = self.compute_saturation_metrics(processed_img)
        cur_snr = self.compute_snr(processed_gray)
        cur_hf_ratio = self.compute_high_freq_ratio(processed_gray)
        
        fidelity = self.compute_fidelity(processed_gray)

        # 2. 硬性约束拦截 (Early Pruning)
        if fidelity < 0.7:
            return -9999.0 
        
        # 截断惩罚：如果处理后出现了大量死黑/死白（例如激增了 5%），直接毙掉
        if (cur_clipping - self.base_clipping) > 0.05:
            return -9999.0

        # 3. 计算相对变化向量 (Deltas)
        # 使用相对变化量保证各指标在同一数量级
        deltas = {
            "sharpness": (cur_sharpness - self.base_sharpness) / (self.base_sharpness + 1e-6),
            "tv": (cur_tv - self.base_tv) / (self.base_tv + 1e-6),
            "entropy": (cur_entropy - self.base_entropy) / (self.base_entropy + 1e-6),
            "brightness": (cur_brightness - self.base_brightness) / (self.base_brightness + 1e-6),
            "saturation": (cur_saturation - self.base_saturation) / (self.base_saturation + 1e-6),
            "snr": (cur_snr - self.base_snr) / (self.base_snr + 1e-6),
            "hf_ratio": (cur_hf_ratio - self.base_hf_ratio) / (self.base_hf_ratio + 1e-6)
        }

        # 4. 任务权重矩阵点积
        reward = sum(task_weights.get(k, 0) * deltas.get(k, 0) for k in deltas.keys())

        # 5. 应用自然度惩罚 (BRISQUE)
        # BRISQUE 越小越好。如果处理后分数变大（恶化），则根据恶化程度施加惩罚
        brisque_degradation = max(0, cur_brisque - self.base_brisque)
        brisque_penalty = brisque_degradation * 0.1  # 惩罚系数可调
        
        final_score = (reward - brisque_penalty) * fidelity

        return float(final_score)