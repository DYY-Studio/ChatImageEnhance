import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import numpy as np

'''
# --------------------------------------------
# Image-Adaptive-3DLUT
# --------------------------------------------
Reference:
@article{zeng2020lut,
  title={Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time},
  author={Zeng, Hui and Cai, Jianrui and Li, Lida and Cao, Zisheng and Zhang, Lei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={04},
  pages={2058--2073},
  year={2022},
  publisher={IEEE Computer Society}
}

@inproceedings{zhang2022clut,
  title={CLUT-Net: Learning Adaptively Compressed Representations of 3DLUTs for Lightweight Image Enhancement},
  author={Zhang, Fengyi and Zeng, Hui and Zhang, Tianjun and Zhang, Lin},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={6493--6501},
  year={2022}
}
'''

# --- 1. 网络基础模块 ---
def discriminator_block(in_filters, out_filters, normalization=False):
    """提取自 models.py 的下采样模块"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class Classifier(nn.Module):
    """提取自 models.py 的图像分类器/权重预测器"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input).squeeze() # 输出尺寸: (3,)

# --- 2. 供应用集成的端到端模型 ---
class Adaptive3DLUT(nn.Module):
    def __init__(self, dim=33):
        super(Adaptive3DLUT, self).__init__()
        self.dim = dim
        self.classifier = Classifier()
        
        # 将3个基础 LUT 注册为不可训练的 buffer
        # 原版包含3个基准LUT，我们将它们初始化为0，稍后由预训练权重覆盖
        self.register_buffer("LUT0", torch.zeros(3, dim, dim, dim))
        self.register_buffer("LUT1", torch.zeros(3, dim, dim, dim))
        self.register_buffer("LUT2", torch.zeros(3, dim, dim, dim))

    def forward(self, img):
        """
        输入: 低分辨率或原始分辨率的图像 Tensor [1, 3, H, W]
        输出: 融合后的 3D LUT Tensor [3, 33, 33, 33]
        """
        # 1. 预测融合权重
        pred_weights = self.classifier(img)
        
        # 2. 线性融合 LUT
        fused_lut = pred_weights[0] * self.LUT0 + \
                    pred_weights[1] * self.LUT1 + \
                    pred_weights[2] * self.LUT2
        
        return fused_lut

def load_adaptive_lut_model(model_dir: str, dim=33, device='cpu'):
    model = Adaptive3DLUT(dim=dim)
    
    # 加载 Classifier 权重
    classifier_weights = torch.load(f"{model_dir}/IA3DLUT_classifier.pth", map_location=device)
    model.classifier.load_state_dict(classifier_weights)
    
    # 加载基础 LUT 权重
    lut_weights = torch.load(f"{model_dir}/IA3DLUT_LUTs.pth", map_location=device)
    # 原版 LUT 模型包含一个叫 "LUT" 的 parameter
    model.LUT0.copy_(lut_weights["0"]["LUT"].squeeze())
    model.LUT1.copy_(lut_weights["1"]["LUT"].squeeze())
    model.LUT2.copy_(lut_weights["2"]["LUT"].squeeze())
    
    model.to(device)
    model.eval()
    return model

def apply_lut_to_image(ori_img_rgb: np.ndarray, input_tensor: torch.Tensor, model, device='cpu'):
    # 2. 获取模型生成的专属 3D LUT
    with torch.inference_mode():
        fused_lut = model(input_tensor) # 形状: [3, 33, 33, 33]

    lut_numpy = fused_lut.permute(1, 2, 3, 0).cpu().numpy()
    
    # 限制范围在 0~1 之间
    lut_numpy = np.clip(lut_numpy, 0.0, 1.0) 
    
    lut_flat = lut_numpy.flatten().tolist() 
    
    dim = model.dim
    pillow_lut = ImageFilter.Color3DLUT(dim, lut_flat)

    ori_img_pil = Image.fromarray(ori_img_rgb)
    result_pil = ori_img_pil.filter(pillow_lut)
    
    return np.array(result_pil)