import cv2
import json
import yaml
import numpy as np
import math
import torch

from PIL import Image
from skimage.metrics import structural_similarity as ssim
from typing import Callable, Any

from models.AestheticScorePredictor import AestheticMLP
from utils import get_executable_dir, get_available_devices

class Evaluator:
    """
    基于弱参考的图像质量评估器。
    """
    def __init__(self, 
        original_img: np.ndarray, 
        model_cache: dict[str, Any] | None = None, 
        device: str | None = None
    ):
        self.original_img = original_img

        self.model_cache = model_cache if model_cache is not None else {}

        if device is None or not device:
            self.device = get_available_devices()[0]
        else:
            self.device = device

        # 预先将原图转为灰度图，节省后续计算 SSIM 的开销
        self.gray_original = self._to_gray(original_img)
        
        # 记录原图的基准指标，可用于计算相对提升比例
        self.base_sharpness = self.compute_sharpness(self.gray_original)
        self.base_tv = self.compute_tv(self.gray_original)
        self.base_entropy = self.compute_entropy(self.gray_original)
        self.base_clipping = self.compute_clipping(self.gray_original)
        self.base_saturation = self.compute_saturation(self.original_img)
        # self.base_color_cast = self.compute_color_cast(self.original_img)
        self.base_snr = self.compute_snr(self.gray_original)
        self.base_hf_ratio = self.compute_high_freq_ratio(self.gray_original)
        self.base_contrast = self.compute_contrast(self.gray_original)

        self.base_lpips_tensor = self._bgr_to_tensor_lpips(self.original_img)
        
        self.brisque_obj = cv2.quality.QualityBRISQUE.create(
            "brisque_model_live.yml", 
            "brisque_range_live.yml"
        ) if hasattr(cv2, "quality") else None

        self.base_brisque = self.compute_brisque(self.original_img)
        self.base_brightness = self.compute_brightness(self.gray_original)

        self.profile_dict = self._generate_profile()

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """安全地将图像转换为灰度图。"""
        if len(img.shape) == 3:
            # 假设输入为 BGR 格式
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    # =========================
    #     深度学习辅助转换
    # =========================
    
    def _bgr_to_pil(self, img: np.ndarray) -> Image.Image:
        """将 OpenCV 的 BGR 格式安全转换为 PIL RGB 格式"""
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _bgr_to_tensor_lpips(self, img: np.ndarray) -> torch.Tensor:
        """将 BGR 格式转换为 LPIPS 需要的范围 [-1, 1] 的 RGB Tensor"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor * 2.0 - 1.0
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _align_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """防御性编程：确保评估时图像尺寸一致，防止 Agent 改变了输出尺寸导致报错。"""
        if img1.shape[:2] != img2.shape[:2]:
            return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        return img2
    
    def _compute_diff(self, 
        func: Callable[[np.ndarray], float], 
        img: np.ndarray, 
        base: float,
        limit: float = 5.0, 
        noise_floor: float = 1e-4
    ) -> float:
        img = self._align_images(self.gray_original, img)
        raw_ratio = (func(img) - base) / (base + noise_floor)
        smoothed_ratio = limit * math.tanh(raw_ratio / limit)
        return smoothed_ratio
    
    # =========================
    #       客观质量指标
    # =========================

    def compute_brightness(self, img: np.ndarray) -> float:
        gray = self._to_gray(img)
        return float(np.mean(gray))
    
    def compare_brightness(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_brightness, img, self.base_brightness)

    def compute_sharpness(self, img: np.ndarray) -> float:
        """
        计算清晰度：拉普拉斯方差 (Variance of Laplacian)。
        值越大，代表图像高频边缘信息越丰富。
        """
        gray = self._to_gray(img)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def compare_sharpness(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_sharpness, img, self.base_sharpness)
    
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
    
    def compare_tv(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_tv, img, self.base_tv)
    
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
    
    def compare_entropy(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_entropy, img, self.base_entropy)

    def compute_clipping(self, img: np.ndarray) -> float:
        """
        计算像素溢出率 (Clipping Ratio)。
        统计绝对纯黑 (0) 和纯白 (255) 像素的占比，用于严厉惩罚极端曝光调整。
        """
        gray = self._to_gray(img)
        # 统计极值像素比例
        clip_ratio = np.mean((gray == 0) | (gray == 255))
        return float(clip_ratio)
    
    def compare_clipping(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_clipping, img, self.base_clipping)

    def compute_brisque(self, img: np.ndarray) -> float:
        """
        无参考图像质量评估 (BRISQUE)。
        分数通常在 0-100 之间，分数越低代表图像感知质量越好（伪影和失真越少）。
        此处取反，分数越高代表图像感知质量越好
        依赖 opencv-contrib-python。
        """
        if self.brisque_obj:
            return 100.0 - self.brisque_obj.compute(img)[0]
        else:
            return 50.0
        
    def compare_brisque(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_brisque, img, self.base_brisque)
        
    def compute_saturation(self, img: np.ndarray) -> float:
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
    
    def compare_saturation(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_saturation, img, self.base_saturation)
    
    def compute_contrast(self, img: np.ndarray) -> float:
        """
        计算全局对比度：均方根对比度 (RMS Contrast)。
        使用灰度图的像素标准差来衡量。值越大，代表对比度越高。
        """
        gray = self._to_gray(img).astype(np.float32)
        # NumPy 的 std 函数直接计算标准差，完美对应 RMS 对比度
        contrast = np.std(gray)
        return float(contrast)
    
    def compare_contrast(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_contrast, img, self.base_contrast)

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
    
    def compare_snr(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_snr, img, self.base_snr)

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
    
    def compare_high_freq_ratio(self, img: np.ndarray) -> float:
        return self._compute_diff(self.compute_high_freq_ratio, img, self.base_hf_ratio)

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
    
    def compute_mse(self, img: np.ndarray) -> float:
        """
        [新增] 计算均方误差 (MSE)。
        用于严格限制像素级别的改变。值越小，与原图越接近。
        """
        gray = self._to_gray(img)
        gray = self._align_images(self.gray_original, gray)
        # 转换为 float64 防止溢出
        mse = np.mean((self.gray_original.astype(np.float64) - gray.astype(np.float64)) ** 2)
        return float(mse)

    def compute_color_shift(self, img: np.ndarray) -> float:
        """
        [新增] 感知色差偏移量计算。
        将图像转换至 LAB 色彩空间计算欧氏距离，这比直接算 RGB 差异更能反映人类视觉对色偏的感知。
        值越低代表偏色越少。
        """
        if len(img.shape) != 3 or len(self.original_img.shape) != 3:
            return 0.0  # 如果有灰度图，跳过色彩偏移检查
            
        img_aligned = self._align_images(self.original_img, img)
        
        # LAB 空间计算距离更符合人类感官
        lab_orig = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_new = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 计算 L, A, B 三通道的欧氏距离
        diff = np.sqrt(np.sum((lab_orig - lab_new) ** 2, axis=-1))
        return float(np.mean(diff))
    
    # =========================
    #     深度感知与语义指标
    # =========================

    def _load_lpips(self, net: str = 'alex'):
        import lpips
        cache_key = f"lpips_{net}_{self.device}"
        # 懒加载：仅在第一次调用此网络时实例化
        if cache_key not in self.model_cache:
            unloads = []
            for key in self.model_cache:
                if key.startswith('lpips_'):
                    unloads.append(key)
            for key in unloads:
                del self.model_cache[key]

            self.model_cache[cache_key] = lpips.LPIPS(net=net).to(self.device).eval()

        return self.model_cache[cache_key]

    def compute_lpips(self, img: np.ndarray, net: str = 'alex') -> float:
        """
        计算 LPIPS 感知相似度。值越低代表感知上越相似。
        """
        # 计算并返回分数
        img1 = self._align_images(self.original_img, img)
        img1 = self._bgr_to_tensor_lpips(img1)
        
        with torch.no_grad():
            score = self._load_lpips(net)(self.base_lpips_tensor, img1).item()
        return 1.0 - np.clip(float(score), 0.0, 1.0)
    
    def _load_clip(self, 
        cache_prefix: str = 'clip', 
        model_name: str = 'ViT-B-32', 
        pretrained: str = 'laion2b_s34b_b79k',
        tokenizer: bool = True,
    ):
        import open_clip
        cache_key = f"{cache_prefix}_{model_name}_{pretrained}_{self.device}"
        
        # 懒加载：将 model, preprocess 和 tokenizer 打包缓存
        if cache_key not in self.model_cache:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device,
                cache_dir=str(get_executable_dir() / 'caches/model_assets/huggingface')
            )
            tokenizer = open_clip.get_tokenizer(model_name) if tokenizer else None

            # 卸载之前的缓存
            unloads = []
            for key in self.model_cache:
                if key.startswith(f'{cache_prefix}_'):
                    unloads.append(key)
            for key in unloads:
                del self.model_cache[key]

            self.model_cache[cache_key] = {
                "model": model.eval(),
                "preprocess": preprocess,
                "tokenizer": tokenizer
            }

        return self.model_cache[cache_key]

    def compute_clip_score(self, 
        img: np.ndarray, 
        text_prompt: str, 
        model_name: str = 'ViT-L-14', 
        pretrained: str = 'openai'
    ) -> float:
        """
        计算图像与文本提示的 CLIP 余弦相似度。值越高代表越符合文本描述。
        支持动态切换 CLIP 模型。
        """
        clip_bundle = self._load_clip('clip', model_name, pretrained)
        model = clip_bundle["model"]
        preprocess = clip_bundle["preprocess"]
        tokenizer = clip_bundle["tokenizer"]

        pil_img = self._bgr_to_pil(img)
        image_input = preprocess(pil_img).unsqueeze(0).to(self.device)
        text_input = tokenizer([text_prompt]).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # 归一化后算余弦相似度
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            
        return float(similarity)

    def compute_aesthetic_score(self, 
        img: np.ndarray, 
        weight_path: str = "sac+logos+ava1-l14-linearMSE.pth", 
        model_name: str = 'ViT-L-14', 
        pretrained: str = 'openai'
    ) -> float:
        """
        计算图像的无参考美学评分 (通常在 1-10 之间)。越高越好。
        注：官方权重库通常基于 ViT-L-14 提取特征，请确保 weight_path 路径正确。
        """
        # 美学打分器需要前置的 CLIP 模型来提取特征

        clip_bundle = self._load_clip('aesthetic_clip', model_name, pretrained)
        model, preprocess = clip_bundle["model"], clip_bundle["preprocess"]

        aes_key = f"aesthetic_mlp_{model_name}"
        if aes_key not in self.model_cache:
            embed_dim = model.visual.output_dim
            mlp = AestheticMLP(embed_dim).to(self.device)
            try:
                # 需提前下载官方权重包到本地路径
                mlp.load_state_dict(torch.load(
                    str(get_executable_dir() / "models" / weight_path), 
                    map_location=self.device
                ))
            except FileNotFoundError:
                print(f"[警告] 美学模型权重 {weight_path} 未找到，返回 0.0")
                return 0.0
            self.model_cache[aes_key] = mlp.eval()
        
        mlp = self.model_cache[aes_key]

        pil_img = self._bgr_to_pil(img)
        image_input = preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            score = mlp(image_features).item()
            
        return float(score)
    
    def preload_models(self,
        model_name: str = 'ViT-L-14',  # ViT-B-32
        pretrained: str = 'openai', # laion2b_s34b_b79k
        l14_pretrained: str = 'openai',
    ):
        loaded = self._load_clip('clip', model_name, pretrained)
        if model_name == 'ViT-L-14' and pretrained == l14_pretrained:
            self.model_cache[f"aesthetic_clip_{model_name}_{pretrained}"] = loaded
        else:
            self._load_clip('aesthetic_clip', 'ViT-L-14', l14_pretrained)
        self._load_lpips()
    
    def _generate_profile(self) -> dict:
        """
        提取图像的尺寸、亮度、对比度、清晰度和噪声估算等关键指标。
        """
        h, w = self.original_img.shape[:2]
        channels = self.original_img.shape[2] if len(self.original_img.shape) == 3 else 1
        
        gray = self.gray_original
        
        # 亮度和对比度计算
        mean_intensity = self.base_brightness
        std_intensity = self.base_contrast
        
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
                    " laplacian_variance": round(self.base_sharpness, 2)
                },
                "quality_metrics": {
                    "information_entropy": round(self.base_entropy, 2),
                    "100.0 - brisque_score": round(self.base_brisque, 2),
                    "clipping_ratio": round(self.base_clipping, 4)
                },
                "color_and_saturation": {
                    "mean_saturation": round(self.base_saturation, 2),
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