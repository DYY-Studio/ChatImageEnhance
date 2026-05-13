import torch, warnings, os
from pathlib import Path
from piq import psnr, ssim, multi_scale_ssim
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

# 获取当前文件所在目录（deeplearning目录）
DEEPLEARNING_DIR = os.path.dirname(os.path.abspath(__file__))

CFG = {
    "input_dir": os.path.join(DEEPLEARNING_DIR, "input_images"),  # 待处理的低清图像目录（支持任意尺寸输入）
    "output_dir": os.path.join(DEEPLEARNING_DIR, "output_images"),  # 高清化后的图像输出目录（保持原文件名）
    "gt_dir": os.path.join(DEEPLEARNING_DIR, "gt_images"),  # (可选) 原始高清参考图目录，用于自动计算质量指标
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

for d in [CFG["input_dir"], CFG["output_dir"], CFG["gt_dir"]]:
    os.makedirs(d, exist_ok=True)

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

class Evaluator:
    def __init__(self):
        self.dev = torch.device(CFG["device"])

    def metrics(self, img_a, img_b):
        """
        计算图像质量核心指标：峰值信噪比(PSNR)、结构相似性(SSIM)、多尺度结构相似性(MS-SSIM)
        所有指标均基于像素级对比，数值越高表示图像质量越好
        :param img_a: 处理后的图像
        :param img_b: 参考高清图像
        :return: 保留4位小数的(PSNR, SSIM, MS-SSIM)元组
        """
        def process_input(img):
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
                tensor = img_transform(img).unsqueeze(0).to(self.dev)
            elif isinstance(img, torch.Tensor):
                tensor = img.unsqueeze(0) if img.dim() == 3 else img
                tensor = tensor.to(self.dev)
            else:
                raise TypeError(f"不支持的输入类型: {type(img)}")
            return tensor

        with torch.no_grad():
            a_tensor = process_input(img_a)
            b_tensor = process_input(img_b)

            # 自动对齐两张图像的尺寸，确保指标计算有效
            if a_tensor.shape != b_tensor.shape:
                import torch.nn.functional as F
                a_tensor = F.interpolate(a_tensor, size=b_tensor.shape[-2:], mode='bilinear', align_corners=False)

            p = psnr(a_tensor, b_tensor, data_range=1.0).item()
            s = ssim(a_tensor, b_tensor, data_range=1.0).item()
            ms = multi_scale_ssim(a_tensor, b_tensor, data_range=1.0).item()
        return round(p, 4), round(s, 4), round(ms, 4)