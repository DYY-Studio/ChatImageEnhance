import cv2
import numpy as np

from pathlib import Path

# 编写高度容错的图像处理函数
# 除了cv库之外也可以考虑引入小参数的传统图像增强模型
# 注意：img必须是第一个参数
def safe_denoise_bilateral(img: np.ndarray, d: int = 5, sigma_color: float = 10.0, sigma_space: float = 10.0) -> np.ndarray:
    try:
        if img is None:
            raise ValueError("Error: Input image is None")

        # 1. 智能处理数据类型
        # original_dtype = img.dtype
        working_img = img.copy()

        d = max(1, int(d))
        sigma_color = float(sigma_color)
        sigma_space = float(sigma_space)

        if working_img.dtype != np.uint8:
            # 如果是 0-1 范围的浮点数，先放大到 0-255
            if working_img.max() <= 1.01: 
                working_img = (working_img * 255).astype(np.uint8)
            else:
                working_img = np.clip(working_img, 0, 255).astype(np.uint8)

        # 2. 针对 3 通道图像，在 Lab 空间进行滤波（效果更好）
        if len(working_img.shape) == 3:
            if working_img.shape[2] == 4:  # BGRA
                working_img = cv2.cvtColor(working_img, cv2.COLOR_BGRA2BGR)
            
            # 核心改进：转换到 Lab 颜色空间
            lab = cv2.cvtColor(working_img, cv2.COLOR_BGR2Lab)
            # 在 Lab 空间应用双边滤波
            denoised_lab = cv2.bilateralFilter(lab, d, sigma_color, sigma_space)
            # 转回 BGR
            result = cv2.cvtColor(denoised_lab, cv2.COLOR_Lab2BGR)
        else:
            # 单通道 Gray 直接处理
            result = cv2.bilateralFilter(working_img, d, sigma_color, sigma_space)

        # 3. 还原回原始数据类型（可选）
        # if original_dtype == np.float32 or original_dtype == np.float64:
        #     return result.astype(original_dtype) / 255.0
        
        return result

    except Exception as e:
        raise RuntimeError(f"BilateralFilter failed: {str(e)}")

def safe_enhance_clahe(img: np.ndarray, clip_limit: float = 1.0, tile_grid_size: int = 4) -> np.ndarray:
    """自适应直方图均衡。注意：内部需自动转换 LAB 色彩空间处理亮度。"""
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # 初始化 CLAHE 对象
        grid = int(tile_grid_size)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))

        if len(img.shape) == 2:
            # 灰度图直接处理
            return clahe.apply(img)
        else:
            # 彩色图：BGR -> LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        raise RuntimeError(f"CLAHE failed: {str(e)}")

# 示范函数
def safe_gaussian_blur(img: np.ndarray, ksize: int = 1) -> np.ndarray:
    """
    【对LLM友好的算子】应用高斯模糊进行降噪。

    事先处理了 ksize 必须为奇数的硬性规定。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        # 1. 防呆：确保输入的是整数
        ksize = int(ksize)
        
        # 2. 防呆：OpenCV 要求高斯核大小必须是正奇数 (1, 3, 5, 7...)
        if ksize % 2 == 0:
            ksize += 1
        if ksize < 1:
            ksize = 1
            
        # 3. 如果是 1，等于不操作，直接原图返回
        if ksize == 1:
            return img.copy()
            
        # 4. 执行真正的 OpenCV 算子
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    except Exception as e:
        # 绝不让进程崩溃，而是返回错误信息供后续逻辑处理
        raise RuntimeError(f"GaussianBlur failed: {str(e)}")
    
def safe_unsharp_masking(img: np.ndarray, amount: float = 1.5, threshold: int = 0) -> np.ndarray:
    """
    反锐化掩模 (Unsharp Masking)。
    比单纯的拉普拉斯锐化更自然，通过增加边缘对比度使图像更清晰。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        # 1. 产生模糊版本作为基准
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        # 2. 图像减去模糊版本获取细节层
        sharpened = float(amount + 1) * img.astype(np.float32) - float(amount) * blurred.astype(np.float32)
        # 3. 截断与转换
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        if threshold > 0:
            low_contrast_mask = np.abs(img.astype(np.int16) - blurred.astype(np.int16)) < threshold
            np.putmask(sharpened, low_contrast_mask, img)
            
        return sharpened
    except Exception as e:
        raise RuntimeError(f"Unsharp masking failed: {str(e)}")
    
def safe_laplacian_sharpening(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    拉普拉斯锐化 (Laplacian Sharpening)。
    利用二阶导数提取图像边缘和高频细节，然后按比例叠加回原图中。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # 1. 计算拉普拉斯算子 (使用更高精度的 float32 避免溢出)
        laplacian = cv2.Laplacian(img, cv2.CV_32F)
        
        # 2. 原图减去(或加上，取决于核的符号)二阶导数细节
        # OpenCV 默认的 Laplacian 核中心为负，所以此处用减法
        sharpened = img.astype(np.float32) - scale * laplacian
        
        # 3. 截断与转换
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened
    except Exception as e:
        raise RuntimeError(f"Laplacian sharpening failed: {str(e)}")
    
def safe_kernel_sharpening(img: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    自定义卷积核锐化 (Kernel Sharpening)。
    使用经典的 3x3 高通滤波掩膜直接对图像进行空间滤波计算。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # 1. 定义经典的中心增强卷积核 (总和为1，保证整体亮度不剧变)
        kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]], dtype=np.float32)
        
        # 2. 进行二维卷积
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 3. 如果强度不是 1.0，则与原图进行线性混合以控制锐化程度
        if intensity != 1.0:
            sharpened = cv2.addWeighted(sharpened, intensity, img, 1.0 - intensity, 0)
            
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Kernel sharpening failed: {str(e)}")
    
def safe_auto_canny(img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    自动 Canny 边缘检测。
    根据图像像素的中位数自动计算高低阈值，无需手动调参。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 计算中位数
        v = np.median(gray)
        # 基于中值生成阈值
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(gray, lower, upper)
    except Exception as e:
        raise RuntimeError(f"Auto Canny failed: {str(e)}")
    
def safe_smart_resize(img: np.ndarray, width: int = None, height: int = None, inter=cv2.INTER_AREA) -> np.ndarray:
    """
    智能缩放（等比例）。
    只需提供宽或高其中一个，另一个会自动计算，避免拉伸变形。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        (h, w) = img.shape[:2]
        if width is None and height is None: return img.copy()

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(img, dim, interpolation=inter)
    except Exception as e:
        raise RuntimeError(f"Resize failed: {str(e)}")

def safe_gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    伽马校正（传统图像增强模型）。
    用于修复过暗或过曝的图片。gamma > 1 变亮，gamma < 1 变暗。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        if gamma <= 0: gamma = 0.1
        # 建立查找表 (LUT) 提高效率
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    except Exception as e:
        raise RuntimeError(f"Gamma correction failed: {str(e)}")
    
def safe_morphology_transform(img: np.ndarray, op_type: str = "open", ksize: int = 3) -> np.ndarray:
    """
    形态学变换 (开运算/闭运算)。
    'open': 先腐蚀后膨胀，用于消除小噪点。
    'close': 先膨胀后腐蚀，用于填充物体内的小孔洞或连接断裂处。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        ksize = max(1, int(ksize))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        
        ops = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "dilate": cv2.MORPH_DILATE,
            "erode": cv2.MORPH_ERODE
        }
        
        op = ops.get(op_type.lower(), cv2.MORPH_OPEN)
        return cv2.morphologyEx(img, op, kernel)
    except Exception as e:
        raise RuntimeError(f"Morphology failed: {str(e)}")

def safe_adaptive_threshold(img: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    自适应二值化。
    相比固定阈值，它能处理光照不均匀的图像（如拍摄的文档）。
    注意：会自动转换为灰度图处理。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        # 预处理：转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 参数纠偏：block_size 必须为大于 1 的奇数
        block_size = int(block_size)
        if block_size % 2 == 0: block_size += 1
        block_size = max(3, block_size)
        
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, int(c)
        )
    except Exception as e:
        raise RuntimeError(f"AdaptiveThreshold failed: {str(e)}")

def safe_median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    中值滤波。
    对抗“椒盐噪声”（图像中随机出现的黑白点）的神器。
    它不像高斯模糊那样会模糊边缘，而是直接剔除离群像素。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        ksize = int(ksize)
        if ksize % 2 == 0: ksize += 1
        ksize = max(1, ksize)
        
        if ksize == 1: return img.copy()
        return cv2.medianBlur(img, ksize)
    except Exception as e:
        raise RuntimeError(f"MedianBlur failed: {str(e)}")

def safe_color_balance(img: np.ndarray, blend_ratio: float = 1.0) -> np.ndarray:
    """
    简单白平衡（灰度世界假设）。
    自动修复照片偏色（例如在暖黄色灯光下拍出的照片变白）。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        if len(img.shape) != 3: return img.copy()
        
        result = img.astype(float)
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        result[:, :, 0] *= (avg_gray / (avg_b + 1e-6))
        result[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
        result[:, :, 2] *= (avg_gray / (avg_r + 1e-6))

        balanced = np.clip(result, 0, 255).astype(np.uint8)
        
        return cv2.addWeighted(balanced, blend_ratio, img, 1.0 - blend_ratio, 0)
    except Exception as e:
        raise RuntimeError(f"ColorBalance failed: {str(e)}")
    
def safe_guided_filter(img: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    """
    引导滤波 (Guided Filter)。
    比双边滤波更快，且彻底解决了边缘处的“光晕”现象。
    常用于：细节增强、图像去噪、抠图边缘平滑。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        # 引导滤波对 float 类型处理更稳健
        img_f = img.astype(np.float32) / 255.0
        # 使用自身作为引导图
        guide = img_f
        
        # 利用 OpenCV 贡献库或通过均值模糊模拟实现
        # 这里使用一种通用实现逻辑
        # 定义窗口大小：使用 2r + 1
        win_size = (int(2 * radius + 1), int(2 * radius + 1))
        
        # 1. 计算均值
        mean_I = cv2.boxFilter(guide, -1, win_size)
        mean_p = cv2.boxFilter(img_f, -1, win_size)
        mean_Ip = cv2.boxFilter(guide * img_f, -1, win_size)

        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(guide * guide, -1, win_size)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, -1, win_size)
        mean_b = cv2.boxFilter(b, -1, win_size)
        
        q = mean_a * guide + mean_b
        return (np.clip(q * 255, 0, 255)).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"GuidedFilter failed: {str(e)}")

def safe_low_light_retinex(img: np.ndarray, sigma_list: list = [15, 80, 250]) -> np.ndarray:
    """
    多尺度 Retinex 增强 (MSR)。
    核心思想是将图像分解为“入射光”和“反射物体”，消除不均匀光照。
    适用于：极暗环境下拍摄的照片，能实现类似“开灯”的效果。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        img_f = img.astype(np.float32) + 1.0 # 防止 log(0)
        msr = np.zeros_like(img_f)
        
        for sigma in sigma_list:
            # 计算单尺度 Retinex
            blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
            msr += np.log10(img_f) - np.log10(blur)
        
        msr /= len(sigma_list)
        
        channels = img.shape[2] if img.ndim == 3 else 1
        
        if channels > 1:
            for i in range(channels):
                c_slice = msr[:, :, i]
                msr[:, :, i] = (c_slice - np.min(c_slice)) / (np.max(c_slice) - np.min(c_slice) + 1e-6) * 255
        else:
            msr = (msr - np.min(msr)) / (np.max(msr) - np.min(msr) + 1e-6) * 255
            
        return msr.astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Retinex failed: {str(e)}")
    
def safe_dehaze(img: np.ndarray, window_size: int = 15, omega: float = 0.95, t0: int = 0.1, guided_radius: int = 40, eps: float = 0.001):
    """
    通用去雾函数 (兼容灰度图和BGR图)
    
    :param img: 输入图像 (uint8)
    :param window_size: 暗通道窗口大小
    :param omega: 去雾强度 (0-1)，通常取 0.95
    :param t0: 透射率最小阈值，防止过度修复
    :param guided_radius: 导向滤波半径
    :param eps: 导向滤波平滑参数
    :return: 去雾后的图像
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        # 1. 预处理：归一化到 [0, 1]
        img = img.astype('float64') / 255.0
        
        # 2. 计算暗通道
        # 如果是彩色图，取三个通道中的最小值；如果是灰度图，直接使用原图
        if len(img.shape) == 3:
            min_channel = np.min(img, axis=2)
        else:
            min_channel = img
        
        # 最小值滤波得到暗通道
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark_channel = cv2.erode(min_channel, kernel)

        # 3. 估算全球大气光值 A
        # 在暗通道中找最亮的 0.1% 像素，对应原图中这些位置的最亮值作为 A
        num_pixels = dark_channel.size
        num_brightest = max(1, int(num_pixels * 0.001))
        dark_vec = dark_channel.reshape(num_pixels)
        indices = dark_vec.argsort()[-num_brightest:]
        
        if len(img.shape) == 3:
            img_vec = img.reshape(num_pixels, 3)
            # 取这些位置中原始像素值最大的作为 A (1x3 向量)
            A = np.max(img_vec[indices], axis=0)
        else:
            img_vec = img.reshape(num_pixels)
            A = np.max(img_vec[indices])

        # 4. 估算透射率 t(x)
        # 这里的计算公式为: t(x) = 1 - omega * min(I/A)
        if len(img.shape) == 3:
            t = 1.0 - omega * cv2.erode(img / A, kernel).min(axis=2)
        else:
            t = 1.0 - omega * cv2.erode(img / A, kernel)

        # 5. 透射率精修 (导向滤波)
        # 这一步能有效消除物体边缘的“白边”或光晕
        t_refined = safe_guided_filter(
            img=t.astype(np.float32), 
            radius=guided_radius, 
            eps=eps
        )

        # 6. 恢复无雾图像 J(x) = (I(x) - A) / max(t(x), t0) + A
        # 扩展 t 的维度以便于广播计算
        if len(img.shape) == 3:
            t_refined_3d = cv2.merge([t_refined, t_refined, t_refined])
            result = (img - A) / np.maximum(t_refined_3d, t0) + A
        else:
            result = (img - A) / np.maximum(t_refined, t0) + A

        # 7. 后处理：限制范围并转回 uint8
        result = np.clip(result * 255, 0, 255).astype('uint8')
        return result
    except Exception as e:
        raise RuntimeError(f"Dehaze failed: {str(e)}")
    
def safe_deringing(img: np.ndarray, threshold: int = 30, blur_sigma: int = 20, window_size: int = 9):
    """
    自适应去环滤波函数 (兼容灰度图与BGR图)
    :param img: 输入图像 (uint8)
    :param threshold: 边缘检测阈值，越小捕捉的伪影区域越多
    :param blur_sigma: 滤波强度（双边滤波的颜色空间标准差）
    :param window_size: 过滤窗口大小
    :return: 去环后的图像
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        # 1. 准备灰度版用于边缘提取
        is_color = len(img.shape) == 3
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img

        # 2. 提取边缘掩码 (使用 Canny 或 Sobel)
        # 振铃通常出现在强边缘附近，我们检测并膨胀这些边缘
        edges = cv2.Canny(gray, threshold, threshold * 3)
        
        # 膨胀掩码以覆盖边缘周围的“环影”区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        mask = cv2.dilate(edges, kernel)
        mask = cv2.GaussianBlur(mask, (window_size, window_size), 0)
        mask_normalized = mask.astype(float) / 255.0

        # 3. 应用强力边缘保护平滑 (双边滤波)
        # 双边滤波在平滑噪声的同时能极好地保持边缘清晰度
        if is_color:
            # 对彩色图进行双边滤波
            filtered = cv2.bilateralFilter(img, window_size, blur_sigma, window_size)
            # 将掩码扩展到 3 通道
            mask_3d = cv2.merge([mask_normalized] * 3)
            # 根据掩码进行线性插值混合：Result = Filtered * Mask + Original * (1 - Mask)
            result = img.astype(float) * (1.0 - mask_3d) + filtered.astype(float) * mask_3d
        else:
            # 对灰度图处理
            filtered = cv2.bilateralFilter(img, window_size, blur_sigma, window_size)
            result = img.astype(float) * (1.0 - mask_normalized) + filtered.astype(float) * mask_normalized

        return np.clip(result, 0, 255).astype(np.uint8)
    except Exception as e:
        raise RuntimeError(f"Deringing failed: {str(e)}")

def safe_hsv_saturation(img: np.ndarray, saturation_scale: float = 1.2) -> np.ndarray:
    """
    自适应饱和度调整。
    通过转换到 HSV 空间，仅增强色彩的鲜艳度而不改变明暗，让图片看起来更“生动”。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        if len(img.shape) != 3: return img.copy() # 灰度图跳过
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # 索引 1 是饱和度通道 (S)
        hsv[:, :, 1] *= saturation_scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    except Exception as e:
        raise RuntimeError(f"Saturation adjustment failed: {str(e)}")
    
def safe_hsv_saturation_nonlinear(img: np.ndarray, saturation_scale: float = 1.2):
    """
    非线性饱和度调整
    通过非线性映射避免过度的饱和度调整，保护图像。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        if len(img.shape) != 3: return img.copy() # 灰度图跳过
        if saturation_scale == 1.0: return img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # 归一化到 0-1 之间进行幂运算
        s = 255 * np.power(s / 255.0, 1.0 / saturation_scale)
        
        s = np.clip(s, 0, 255).astype(np.uint8)
        h = h.astype(np.uint8)
        s = s.astype(np.uint8)
        v = v.astype(np.uint8)
        
        # 4. 合并并转回 BGR
        hsv_final = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    except Exception as e:
        raise RuntimeError(f"Saturation Nonlinear adjustment failed: {str(e)}")
    
def safe_vibrance(img: np.ndarray, amount: int = 0):
    """
    自然饱和度
    只增强那些饱和度原本较低的像素
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")

        if len(img.shape) != 3: return img.copy() # 灰度图跳过
        if amount == 0: return img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # 计算平均饱和度，作为调整参考
        # 只针对饱和度低于平均值的区域进行较大幅度的提升
        vibrance_mask = 1.0 - (s / 255.0) 
        s += amount * vibrance_mask
        
        s = np.clip(s, 0, 255).astype(np.uint8)
        h = h.astype(np.uint8)
        s = s.astype(np.uint8)
        v = v.astype(np.uint8)
        
        # 4. 合并并转回 BGR
        hsv_final = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_final, cv2.COLOR_HSV2BGR)
    except Exception as e:
        raise RuntimeError(f"Vibrance adjustment failed: {str(e)}")

def safe_color_temperature(img: np.ndarray, temperature: float = 0.0, tint: float = 0.0) -> np.ndarray:
    """
    色温与色调微调 (模拟 Lightroom 逻辑)。
    temperature > 0 变暖 (偏黄/红)，< 0 变冷 (偏蓝)。
    tint > 0 偏洋红，< 0 偏绿。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        if len(img.shape) != 3: return img.copy() # 灰度图直接返回
        
        # 转换到 np.int16 以防止加减运算时的溢出
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        
        a += tint  # 调整色调
        b += temperature  # 调整色温
        
        lab = cv2.merge((l, a, b))
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        raise RuntimeError(f"Color temperature adjustment failed: {str(e)}")

def safe_hue_shift(img: np.ndarray, hue_shift: int = 0) -> np.ndarray:
    """
    全局色相平移。
    在 HSV 空间中直接旋转色相轮，用于彻底改变物体的颜色规律或进行艺术化调色。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        if len(img.shape) != 3: return img.copy()
        
        if hue_shift == 0:
            return img.copy()

        # 转换到 HSV 空间
        # OpenCV 中，H 的范围是 0-180 (而不是 0-360)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        
        # 平移 H 通道
        hsv[:, :, 0] += hue_shift
        
        # 处理色相轮的环绕溢出 (Wrap around)
        hsv[:, :, 0] = hsv[:, :, 0] % 180
        
        # 转换回 BGR
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        raise RuntimeError(f"Hue shift failed: {str(e)}")
    
def safe_enhance_detail(img: np.ndarray, sigma_s: float = 10.0, sigma_r: float = 0.15) -> np.ndarray:
    """HDR级细节增强。注意：原生仅支持3通道，灰度图会在内部进行转换。"""
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        sigma_s = float(sigma_s)
        sigma_r = float(sigma_r)
        
        is_gray = (len(img.shape) == 2 or img.shape[2] == 1)
        # cv2.detailEnhance 强制要求 8-bit 3通道
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if is_gray else img
        
        enhanced = cv2.detailEnhance(bgr_img, sigma_s=sigma_s, sigma_r=sigma_r)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if is_gray else enhanced
    except Exception as e:
        raise RuntimeError(f"Detail Enhancement failed: {str(e)}")

def safe_nl_means_denoise(img: np.ndarray, h: float = 10.0, template_window: int = 7, search_window: int = 21) -> np.ndarray:
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # OpenCV 的 NL-Means 函数要求窗口大小必须是奇数
        template_window = int(template_window) if template_window % 2 == 1 else int(template_window) + 1
        search_window = int(search_window) if search_window % 2 == 1 else int(search_window) + 1
        
        # 确保输入是 uint8 类型 (OpenCV 此 API 的严格要求)
        if img.dtype != np.uint8:
            if img.dtype in [np.float32, np.float64] and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # 根据通道数选择不同的 OpenCV API
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            # 单通道灰度图
            denoised = cv2.fastNlMeansDenoising(img, None, h, template_window, search_window)
            
        elif img.ndim == 3 and img.shape[2] == 3:
            # 3通道彩色图 (OpenCV 会将图像转换到 YUV 空间，对亮度通道和颜色通道分别降噪)
            # h 为亮度通道的滤波强度，hColor(第二个h) 为颜色通道的滤波强度，通常设为相等即可
            denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)
            
        elif img.ndim == 3 and img.shape[2] == 4:
            # 4通道带有 Alpha 通道的图像，分离出 RGB 处理，再合并
            rgb_channels = img[:, :, :3]
            alpha_channel = img[:, :, 3]
            denoised_rgb = cv2.fastNlMeansDenoisingColored(rgb_channels, None, h, h, template_window, search_window)
            denoised = np.dstack((denoised_rgb, alpha_channel))
            
        else:
            raise ValueError(f"Unsupported image shape for NL-Means: {img.shape}")
            
        return denoised
        
    except Exception as e:
        raise RuntimeError(f"NL-Means denoising failed: {str(e)}")

# _ZERODCE_NET = None
# def safe_zero_dce(img: np.ndarray) -> np.ndarray:
#     """
#     Zero-DCE 轻量级神经网络零光照度增强
#     只支持三通道图像
#     """
#     global _ZERODCE_NET
#     try:
#         if img is None: raise ValueError("Error: Input image is None")
#         if not Path("models/zerodce_integrated.onnx").exists():
#             raise FileNotFoundError("Error: Model file not found")
        
#         if len(img.shape) != 3: return img.copy()
        
#         if _ZERODCE_NET is None:
#             model_path = "models/zerodce_integrated.onnx"
            
#             _ZERODCE_NET = cv2.dnn.readNetFromONNX(model_path)
#             _ZERODCE_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#             _ZERODCE_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#         blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0, swapRB=True)

#         _ZERODCE_NET.setInput(blob)
#         out_blob = _ZERODCE_NET.forward()

#         output = out_blob.squeeze().transpose(1, 2, 0) # [H, W, 3]
#         output = (output * 255).clip(0, 255).astype(np.uint8)
#         return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
#     except Exception as e:
#         raise RuntimeError(f"ZeroDCE failed: {str(e)}")

if hasattr(cv2, "xphoto") and hasattr(cv2, "ximgproc"):
    # def safe_filter_guided(img: np.ndarray, radius: int = 8, eps: float = 0.01) -> np.ndarray:
    #     """导向滤波 (保边缘平滑)。注意：无梯度翻转现象，常作为双边滤波的上位替代。"""
    #     try:
    #         if img is None: raise ValueError("Error: Input image is None")
            
    #         radius = max(1, int(radius))
    #         eps = float(eps)

    #         return cv2.ximgproc.guidedFilter(img, img, radius, eps)
    #     except Exception as e:
    #         raise RuntimeError(f"GuidedFilter failed: {str(e)}")
    
    def safe_enhance_grayworld_wb(img: np.ndarray) -> np.ndarray:
        """灰度世界白平衡。注意：用于自动色彩校正和去偏色，仅对彩色图有效。"""
        try:
            if img is None: raise ValueError("Error: Input image is None")
            
            # 若为灰度图则无需白平衡，直接返回
            if len(img.shape) == 2 or img.shape[2] != 3:
                return img
                
            # 使用 xphoto 模块的白平衡算法
            wb = cv2.xphoto.createGrayworldWB()
            return wb.balanceWhite(img)
        except Exception as e:
            raise RuntimeError(f"Grayworld WB failed: {str(e)}")
        
    def safe_enhance_sauvola(img: np.ndarray, block_size: int = 15, k: float = 0.2) -> np.ndarray:
        """Sauvola 局部二值化。注意：常用于处理光照不均匀的文档/发票图像增强。"""
        try:
            if img is None: raise ValueError("Error: Input image is None")
            
            # 参数纠偏
            block_size = max(3, int(block_size))
            if block_size % 2 == 0:
                block_size += 1 # 邻域窗口必须为奇数
            k = float(k)

            # 算法强制要求单通道
            gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用 ximgproc 的 niBlackThreshold 实现 Sauvola (Niblack的改进版)
            binary = cv2.ximgproc.niBlackThreshold(
                gray, 
                maxValue=255, 
                type=cv2.THRESH_BINARY, 
                blockSize=block_size, 
                k=k, 
                binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA
            )
            return binary
        except Exception as e:
            raise RuntimeError(f"Sauvola Binarization failed: {str(e)}")
        
    def safe_denoise_anisotropic(img: np.ndarray, alpha: float = 0.1, k: float = 15.0, niters: int = 10) -> np.ndarray:
        """各向异性扩散滤波。注意：底层严格要求 CV_8UC3，内部已做通道和类型的安全兼容。"""
        try:
            if img is None: raise ValueError("Error: Input image is None")
            
            # 参数纠偏
            alpha = float(alpha)
            k = float(k)
            niters = max(1, int(niters))

            # 1. 强制数据类型为 uint8 (满足 CV_8U)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            # 2. 强制通道数为 3 (满足 C3)
            is_gray = False
            if len(img.shape) == 2 or img.shape[2] == 1:
                is_gray = True
                process_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # 丢弃 Alpha 通道，防止底层崩溃
                process_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                process_img = img

            # 3. 执行核心算子 (此时 process_img 必定为 CV_8UC3)
            res = cv2.ximgproc.anisotropicDiffusion(process_img, alpha, k, niters)
            
            # 4. 若原图为灰度图，则恢复单通道状态返回
            if is_gray:
                return cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                
            return res
        except Exception as e:
            raise RuntimeError(f"Anisotropic Diffusion failed: {str(e)}")
        
    def safe_smooth_l0(img: np.ndarray, kappa: float = 2.0, lambda_param: float = 0.02) -> np.ndarray:
        """L0 梯度平滑。注意：常用于抹除杂乱纹理并产生类似插画/色块平滑的效果。"""
        try:
            if img is None: raise ValueError("Error: Input image is None")
            
            kappa = float(kappa)
            lambda_param = float(lambda_param)
            
            # cv2.ximgproc.l0Smooth 完美兼容 1 和 3 通道
            return cv2.ximgproc.l0Smooth(img, kappa=kappa, lambda_=lambda_param)
        except Exception as e:
            raise RuntimeError(f"L0 Smooth failed: {str(e)}")
        
    def safe_filter_rolling_guidance(img: np.ndarray, d: int = -1, sigma_color: float = 25.0, sigma_space: float = 3.0, num_iters: int = 4) -> np.ndarray:
        """滚动导向滤波。注意：极佳的宏观结构保留与微观纹理抹除算子。"""
        try:
            if img is None: raise ValueError("Error: Input image is None")
            
            d = int(d)
            sigma_color = float(sigma_color)
            sigma_space = float(sigma_space)
            num_iters = max(1, int(num_iters))
            
            # 支持 1 和 3 通道
            return cv2.ximgproc.rollingGuidanceFilter(img, None, d, sigma_color, sigma_space, num_iters)
        except Exception as e:
            raise RuntimeError(f"Rolling Guidance Filter failed: {str(e)}")

if __name__ == "__main__":
    # 测试所有CV算子能否正常运作，是否在特定格式图像上会出现问题

    # 测试用随机灰度图像
    gray_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    # 测试用随机彩色图像
    color_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    import inspect
    import sys

    def run_all_func():
        current_module = sys.modules[__name__]
        functions_list = inspect.getmembers(current_module, inspect.isfunction)
        for name, func in functions_list:
            if name == "run_all_func": continue
            if isinstance(ret := func(gray_img), str):
                print("GRAY", ret)
            if isinstance(ret := func(color_img), str):
                print("COLOR", ret)

    run_all_func()