import cv2
import numpy as np
import importlib.util, importlib
from skimage import util, restoration, exposure

def safe_tv_denoise_chambolle(img: np.ndarray, weight: float = 0.1) -> np.ndarray:
    """
    全变分降噪

    TV 降噪是一种极好的保边滤波算法。相比高斯模糊会使得边缘变糊，TV 降噪能在抹平纹理噪点的同时，保持物体边缘的锐利。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        is_color = img.ndim == 3 and img.shape[2] in [3, 4]
        kwargs = {'channel_axis': -1} if is_color else {}
        
        # TV 降噪，输出为 [0.0, 1.0] 的 float64
        denoised = restoration.denoise_tv_chambolle(img, weight=weight, **kwargs)
        
        return util.img_as_ubyte(denoised)
    except Exception as e:
        raise RuntimeError(f"TV Denoising failed: {str(e)}")
    
def safe_tv_denoise_bregman(img: np.ndarray, weight: float = 5.0, isotropic: bool = True) -> np.ndarray:
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # 确定是否为彩色图像以设置 channel_axis
        is_color = img.ndim == 3 and img.shape[2] in [3, 4]
        kwargs = {'channel_axis': -1} if is_color else {}
        
        # skimage 需要浮点型输入以确保数学优化的精度
        img_float = util.img_as_float(img)
        
        # 运行 Split-Bregman 降噪
        denoised = restoration.denoise_tv_bregman(
            img_float, 
            weight=weight, 
            isotropic=isotropic,
            **kwargs
        )
        
        # 裁剪防止溢出，并安全转换回 uint8
        denoised = np.clip(denoised, 0.0, 1.0)
        return util.img_as_ubyte(denoised)
        
    except Exception as e:
        raise RuntimeError(f"TV Bregman Denoising failed: {str(e)}")
    
def safe_adjust_sigmoid(img: np.ndarray, cutoff: float = 0.5, gain: float = 10.0) -> np.ndarray:
    """
    S型曲线对比度调整 (Adjust Sigmoid)

    利用 Sigmoid 函数的 S 型非线性曲线，能够极大地提升图像中间调（Midtones）的对比度
    同时柔和地压缩极亮和极暗区域，防止严重的高光溢出或死黑。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # exposure.adjust_sigmoid 兼容 uint8，但为防意外显式转换
        corrected = exposure.adjust_sigmoid(img, cutoff=cutoff, gain=gain)
        
        if corrected.dtype != np.uint8:
            corrected = util.img_as_ubyte(corrected)
            
        return corrected
    except Exception as e:
        raise RuntimeError(f"Sigmoid adjustment failed: {str(e)}")
    
def safe_richardson_lucy(img: np.ndarray, iterations: int = 15, psf_size: int = 5) -> np.ndarray:
    """
    理查德森-露西去卷积 (Richardson-Lucy Deconvolution)

    一种经典的迭代去模糊算法，通常用于天文摄影或显微成像中的失焦恢复
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # 1. 构造一个近似的点扩散函数 (PSF) —— 这里使用二维高斯核近似常见的光学模糊
        sigma = psf_size / 3.0
        psf1d = cv2.getGaussianKernel(psf_size, sigma)
        psf2d = np.outer(psf1d, psf1d)
        
        # 2. RL 去卷积需要在 [0.0, 1.0] 的浮点精度下运算
        img_float = util.img_as_float(img)
        is_color = img.ndim == 3 and img.shape[2] in [3, 4]
        
        if is_color:
            restored = np.zeros_like(img_float)
            # 必须逐通道进行去卷积，否则 skimage 会把三维图像当成 3D 体素处理导致崩溃
            for i in range(img.shape[2]):
                restored[:, :, i] = restoration.richardson_lucy(
                    img_float[:, :, i], psf2d, num_iter=iterations
                )
        else:
            restored = restoration.richardson_lucy(img_float, psf2d, num_iter=iterations)
        
        # 3. RL 算法有时会导致像素值溢出 [0, 1] 区间，需要做 Clip 后再转换
        restored = np.clip(restored, 0.0, 1.0)
        return util.img_as_ubyte(restored)
        
    except Exception as e:
        raise RuntimeError(f"Richardson-Lucy deconvolution failed: {str(e)}")
    
def safe_unsupervised_wiener(img: np.ndarray, psf_size: int = 5) -> np.ndarray:
    """
    无监督维纳滤波 (Unsupervised Wiener)

    维纳滤波不仅去模糊，还能自动平衡降噪（它会自动寻找图像噪声与信号功率的平衡点），
    因此比 RL 更不容易产生噪点伪影，适合带噪声的模糊图像。
    """
    try:
        if img is None: raise ValueError("Error: Input image is None")
        
        # 构造高斯 PSF
        sigma = psf_size / 3.0
        psf1d = cv2.getGaussianKernel(psf_size, sigma)
        psf2d = np.outer(psf1d, psf1d)
        
        img_float = util.img_as_float(img)
        is_color = img.ndim == 3 and img.shape[2] in [3, 4]
        
        if is_color:
            restored = np.zeros_like(img_float)
            for i in range(img.shape[2]):
                # unsupervised_wiener 返回一个元组: (复原后的图像, 评估出的马尔科夫链参数)
                # 我们只需要复原后的图像 [0]
                deconvolved, _ = restoration.unsupervised_wiener(img_float[:, :, i], psf2d)
                restored[:, :, i] = deconvolved
        else:
            restored, _ = restoration.unsupervised_wiener(img_float, psf2d)
            
        restored = np.clip(restored, 0.0, 1.0)
        return util.img_as_ubyte(restored)
        
    except Exception as e:
        raise RuntimeError(f"Unsupervised Wiener deconvolution failed: {str(e)}")
    
if importlib.util.find_spec('pywt'):
    # 小波降噪函数需要 PyWavelets 库支持
    pywt = importlib.import_module('pywt')
    def safe_wavelet_denoise(img: np.ndarray, levels: int = 3) -> np.ndarray:
        """
        小波降噪

        在频域进行降噪，非常适合处理带有高斯白噪声的图片（例如夜景或高 ISO 拍摄产生的颗粒感噪点）。
        """
        try:
            if img is None: raise ValueError("Error: Input image is None")
            
            is_color = img.ndim == 3 and img.shape[2] in [3, 4]
            kwargs = {'channel_axis': -1} if is_color else {}
            
            # 小波降噪需要输入 float 类型的图像以获得最佳效果
            img_float = util.img_as_float(img)
            
            # 使用 BayesShrink 方法和 soft 模式，通常能得到普适性好的结果
            denoised = restoration.denoise_wavelet(
                img_float, 
                wavelet_levels=levels, 
                method='BayesShrink', 
                mode='soft', 
                rescale_sigma=True, 
                **kwargs
            )
            
            return util.img_as_ubyte(denoised)
        except Exception as e:
            raise RuntimeError(f"Wavelet Denoising failed: {str(e)}")

if __name__ == "__main__":
    # 测试所有CV算子能否正常运作，是否在特定格式图像上会出现问题

    # 测试用随机灰度图像
    # gray_img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)

    # 测试用随机彩色图像
    color_img = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)

    import inspect
    import sys
    import time

    def run_all_func():
        current_module = sys.modules[__name__]
        functions_list = inspect.getmembers(current_module, inspect.isfunction)
        for name, func in functions_list:
            if name == "run_all_func": continue
            # if isinstance(ret := func(gray_img), str):
            #     print("GRAY", ret)
            try:
                print(f"{name}", end="")
                func(color_img)
                start_time = time.perf_counter_ns()
                for _ in range(20):
                    func(color_img)
                end_time = time.perf_counter_ns()
                print(f",{(end_time - start_time) / 20 / 1_000_000 }")
            except Exception as e:
                print(str(e))

    run_all_func()