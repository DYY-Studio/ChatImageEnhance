import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

'''
# --------------------------------------------
# SepLUT
# --------------------------------------------
Reference:
@InProceedings{yang2022seplut,
  title={SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement},
  author={Yang, Canqian and Jin, Meiguang and Xu, Yi and Zhang, Rui and Chen, Ying and Liu, Huaida},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
'''

# ==========================================
# 1. Backbones (特征提取主干网络)
# ==========================================
class BasicBlock(nn.Sequential):
    """基础卷积块 (Conv + LeakyReLU [+ InstanceNorm])"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)

class LightBackbone(nn.Sequential):
    """轻量级 5 层 CNN 主干网络"""
    def __init__(self, input_resolution=256, extra_pooling=False, n_base_feats=8, **kwargs):
        body = [BasicBlock(3, n_base_feats, stride=2, norm=True)]
        n_feats = n_base_feats
        for _ in range(3):
            body.append(BasicBlock(n_feats, n_feats * 2, stride=2, norm=True))
            n_feats = n_feats * 2
        body.append(BasicBlock(n_feats, n_feats, stride=2))
        body.append(nn.Dropout(p=0.5))
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = n_feats * (4 if extra_pooling else (input_resolution // 32) ** 2)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)

class Res18Backbone(nn.Module):
    """ResNet-18 主干网络"""
    def __init__(self, pretrained=False, input_resolution=224, **kwargs):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)


# ==========================================
# 2. LUT Generators (LUT 生成器)
# ==========================================
def lut_transform(imgs, luts):
    """利用原生 F.grid_sample 执行 LUT 变换"""
    # 归一化像素值到 [-1, 1] 范围，以匹配 grid_sample 的坐标要求
    imgs = (imgs - .5) * 2.
    grids = imgs.permute(0, 2, 3, 1).unsqueeze(1)
    
    outs = F.grid_sample(luts, grids, mode='bilinear', padding_mode='border', align_corners=True)
    outs = outs.squeeze(2)
    return outs

class LUT1DGenerator(nn.Module):
    """1D LUT 生成器"""
    def __init__(self, n_colors, n_vertices, n_feats, color_share=False):
        super().__init__()
        repeat_factor = n_colors if not color_share else 1
        self.lut1d_generator = nn.Linear(n_feats, n_vertices * repeat_factor)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.color_share = color_share

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        lut1d = self.lut1d_generator(x).view(x.shape[0], -1, self.n_vertices)
        if self.color_share:
            lut1d = lut1d.repeat_interleave(self.n_colors, dim=1)
        lut1d = lut1d.sigmoid()
        return lut1d

class LUT3DGenerator(nn.Module):
    """3D LUT 生成器"""
    def __init__(self, n_colors, n_vertices, n_feats, n_ranks):
        super().__init__()
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        self.basis_luts_bank = nn.Linear(n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks

    def forward(self, x):
        weights = self.weights_generator(x)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts


class SepLUTGenerator(nn.Module):
    """
    极简版 SepLUT 生成器：只输出 LUT 矩阵，不执行图像变换
    """
    def __init__(self,
                 n_ranks=3,
                 n_vertices_3d=17,
                 n_vertices_1d=17,
                 lut1d_color_share=False,
                 backbone='light',
                 n_base_feats=8,
                 pretrained=False,
                 n_colors=3):
        super().__init__()
        
        if backbone == 'light':
            self.backbone = LightBackbone(extra_pooling=True, n_base_feats=n_base_feats)
        else:
            self.backbone = Res18Backbone(pretrained=pretrained)

        self.n_colors = n_colors
        self.n_vertices_3d = n_vertices_3d
        self.n_vertices_1d = n_vertices_1d

        if n_vertices_3d > 0:
            self.lut3d_generator = LUT3DGenerator(n_colors, n_vertices_3d, self.backbone.out_channels, n_ranks)
        if n_vertices_1d > 0:
            self.lut1d_generator = LUT1DGenerator(n_colors, n_vertices_1d, self.backbone.out_channels, color_share=lut1d_color_share)

    def forward(self, imgs):
        """
        输入: imgs (B, C, H, W) 通常为降采样后的图(如 256x256), 取值 [0,1], RGB顺序
        输出: lut1d (B, 3, M), lut3d (B, 3, D, D, D) 的 numpy 数组形式
        """
        codes = self.backbone(imgs)

        lut1d_np, lut3d_np = None, None
        
        # 1. 生成 1D LUT
        if self.n_vertices_1d > 0:
            lut1d = self.lut1d_generator(codes)
            lut1d_np = lut1d.detach().cpu().squeeze(0).numpy()

        # 2. 生成 3D LUT
        if self.n_vertices_3d > 0:
            _, lut3d = self.lut3d_generator(codes)
            lut3d_np = lut3d.detach().cpu().squeeze(0).numpy()

        return lut1d_np, lut3d_np

import numpy as np
import cv2
from PIL import Image, ImageFilter
def apply_lut1d_opencv(img_bgr, lut1d_np):
    """
    使用 OpenCV 套用 1D LUT。

    :param img_bgr: 原始全尺寸 BGR 图像 (H, W, 3), uint8
    :param lut1d_np: 模型输出的 1D LUT 数组，shape (3, M), 取值 [0, 1]
    :return: 增强后的 BGR 图像
    """
    # 获取顶点数 M (如 17)
    M = lut1d_np.shape[1]
    
    # 构建插值的基准坐标
    x_orig = np.linspace(0, 255, M)
    x_new = np.arange(256)
    
    # 建立供 cv2.LUT 使用的 256 元素映射表 (256, 1, 3)
    # 注意: 模型输出的 LUT 通常基于 RGB 通道，而 img_bgr 是 BGR
    # 所以要将 PyTorch 的 R, G, B LUT 对应映射给 BGR 图像的 2, 1, 0 通道
    lut_256 = np.zeros((256, 1, 3), dtype=np.uint8)
    
    # R 通道 (对应模型 0, 图像 2)
    lut_256[:, 0, 2] = np.clip(np.interp(x_new, x_orig, lut1d_np[0] * 255.0), 0, 255).astype(np.uint8)
    # G 通道 (对应模型 1, 图像 1)
    lut_256[:, 0, 1] = np.clip(np.interp(x_new, x_orig, lut1d_np[1] * 255.0), 0, 255).astype(np.uint8)
    # B 通道 (对应模型 2, 图像 0)
    lut_256[:, 0, 0] = np.clip(np.interp(x_new, x_orig, lut1d_np[2] * 255.0), 0, 255).astype(np.uint8)

    # 利用 cv2 极速查表替换像素
    return cv2.LUT(img_bgr, lut_256)


def apply_lut3d_pil(img_rgb_np, lut3d_np):
    """
    使用 Pillow 极速套用 3D LUT
    
    :param img_rgb_np: 全尺寸 RGB 图像, numpy uint8
    :param lut3d_np: 模型输出的 3D LUT，shape为 (3, D, D, D), 取值 [0, 1]
    :return: 最终增强的 RGB 图像 numpy 数组
    """
    D = lut3d_np.shape[1] # 网格大小，通常为 17

    # 1. 修正维度转置：
    # PyTorch 输出的 lut3d_np shape 为 (C, Blue, Green, Red)
    # Pillow 要求展平后的循环顺序是: 外层B, 中层G, 内层R (R变化最快)
    # 所以我们需要把维度变为 (Blue, Green, Red, C)
    lut_transposed = np.transpose(lut3d_np, (1, 2, 3, 0))
    
    # 2. 修正数值范围：
    # Pillow 期望浮点数在 [0.0, 1.0] 之间，千万不能乘以 255！
    lut_flat = np.clip(lut_transposed, 0.0, 1.0).flatten().tolist()
    
    # 构建 Pillow 3DLUT 滤波器
    filter_3dlut = ImageFilter.Color3DLUT((D, D, D), lut_flat)
    
    # 转换为 PIL Image 并套用
    img_pil = Image.fromarray(img_rgb_np)
    img_filtered = img_pil.filter(filter_3dlut)
    
    return np.array(img_filtered)