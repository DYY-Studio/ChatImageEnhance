import torch
import torch.nn as nn

"""
# --------------------------------------------
# SCI
# --------------------------------------------
Reference:
@ARTICLE{11072373,
  author={Ma, Long and Ma, Tengyu and Xu, Chengpei and Liu, Jinyuan and Fan, Xin and Luo, Zhongxuan and Liu, Risheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning With Self-Calibrator for Fast and Robust Low-Light Image Enhancement}, 
  year={2025},
  volume={47},
  number={10},
  pages={9095-9112}
}
"""

def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size//2), bias=bias)

class EnhanceNetwork_Ha(nn.Module):
    """
    轻量级光照估计子网络
    """
    def __init__(self, layers=1, channels=3):
        super(EnhanceNetwork_Ha, self).__init__()

        kernel_size = 3
        self.in_conv = nn.Sequential(
            default_conv(dim_in=3, dim_out=channels, kernel_size=kernel_size, bias=True),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for _ in range(layers):
            conv = nn.Sequential(
                default_conv(dim_in=channels, dim_out=channels, kernel_size=kernel_size, bias=True),
                nn.ReLU()
            )
            self.blocks.append(conv)

        self.out_conv = nn.Sequential(
            default_conv(dim_in=channels, dim_out=3, kernel_size=kernel_size, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.0001, 1) # 防止除零错误
        return illu

class SCIRuntime(nn.Module):
    def __init__(self):
        super(SCIRuntime, self).__init__()
        # 对应原文中 stage=1 的极简配置
        self.ha = EnhanceNetwork_Ha(layers=1, channels=3)

    def forward(self, input):
        # 1. 估算光照图 i
        i = self.ha(input)
        # 2. 基于 Retinex 原理，去除光照得到反射率 r (即增强图)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return r