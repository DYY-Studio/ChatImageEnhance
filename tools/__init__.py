# 注册函数
from tools.registry import ToolRegistry
from tools.cv_wrappers import *
from tools.skimage_wrappers import *

# 1. 实例化注册表
global_registry = ToolRegistry()

# 2. 核心操作：向系统注册函数
# 1. 双边滤波注册
global_registry.register(
    name="Bilateral_Filter",
    func=safe_denoise_bilateral,
    description="双边滤波去噪。在平滑图像的同时能够很好地保留边缘特征，不会像高斯模糊那样让物体边界变虚。适用于需要保边降噪的预处理。",
    params_schema={
        "d": {
            "type": "int",
            "range": [1, 9],
            "description": "过滤过程中每个像素邻域的直径。值越大，参与计算的区域越广。建议取值 5 或 9。"
        },
        "sigma_color": {
            "type": "float",
            "range": [10.0, 150.0],
            "description": "颜色空间滤波器的标准差。值越大，表示邻域内越宽广的颜色会被混合在一起，产生半透明的颜色抹匀效果。"
        },
        "sigma_space": {
            "type": "float",
            "range": [10.0, 150.0],
            "description": "坐标空间中滤波器的标准差。值越大，只要颜色足够接近，越远的像素就会相互影响。"
        }
    }
)

# 2. 自适应直方图均衡注册
global_registry.register(
    name="CLAHE_Enhancement",
    func=safe_enhance_clahe,
    description="自适应直方图均衡化。专门用于提升图像对比度，能让暗部细节显现，同时防止过曝。已内置色彩空间转换，支持彩色图。",
    params_schema={
        "clip_limit": {
            "type": "float",
            "range": [1.0, 4.0],
            "description": "对比度限制阈值。值越高对比度提升越明显，但可能会放大噪点。默认 2.0。"
        },
        "tile_grid_size": {
            "type": "int",
            "range": [4, 16],
            "description": "网格大小。将图像划分为 M x M 的区域进行局部均衡。建议取 8。"
        }
    }
)

# 3. 伽马校正注册
global_registry.register(
    name="Gamma_Correction",
    func=safe_gamma_correction,
    description="调整图像曝光度。Gamma > 1 整体变亮（修复暗光），Gamma < 1 整体变暗（修复过曝）。比线性亮度调整更符合人眼视觉。",
    params_schema={
        "gamma": {
            "type": "float",
            "range": [0.1, 3.0],
            "description": "校正系数。1.0 为原图；建议暗光环境搜索 [1.2, 2.2]，强光环境搜索 [0.4, 0.8]。"
        }
    }
)

# 4. 反锐化掩模注册
global_registry.register(
    name="Unsharp_Masking",
    func=safe_unsharp_masking,
    description="高级锐化算子。通过增强边缘对比度使图像看起来更清晰，比普通锐化更自然。适用于修复轻微失焦的图片。",
    params_schema={
        "amount": {
            "type": "float",
            "range": [0.5, 3.0],
            "description": "锐化强度。值越高，边缘越尖锐。默认 1.5。"
        },
        "threshold": {
            "type": "int",
            "range": [0, 10],
            "description": "锐化阈值。只有相邻像素差值大于此值才进行锐化，设为 0 表示全图锐化处理。"
        }
    }
)

global_registry.register(
    name="Laplacian_Sharpening",
    func=safe_laplacian_sharpening,
    description="二阶导数锐化算子。直接提取图像的细微纹理和高频边缘并叠加到原图，适合增强非常细微的细节。",
    params_schema={
        "scale": {
            "type": "float",
            "range": [0.1, 3.0],
            "description": "细节叠加系数。值越大，高频细节越明显。默认 1.0。"
        }
    }
)

global_registry.register(
    name="Kernel_Sharpening",
    func=safe_kernel_sharpening,
    description="基础空间滤波锐化。通过 3x3 卷积核增强中心像素与周围像素的对比度。计算速度极快。",
    params_schema={
        "intensity": {
            "type": "float",
            "range": [0.0, 2.0],
            "description": "锐化结果的混合比例。1.0 为纯锐化结果，0.0 为原图，介于两者之间为平滑过渡。"
        }
    }
)

# 5. 自动 Canny 边缘检测注册
global_registry.register(
    name="Auto_Canny",
    func=safe_auto_canny,
    description="自动边缘检测。基于图像统计学特征自动确定高低阈值，输出二值化的边缘轮廓图。适用于轮廓提取。",
    params_schema={
        "sigma": {
            "type": "float",
            "range": [0.2, 0.5],
            "description": "阈值灵敏度。较小的值会产生更严格的边缘（边缘更少），较大的值会产生更丰富的边缘。默认 0.33。"
        }
    }
)

# 6. 智能缩放注册
# global_registry.register(
#     name="Smart_Resize",
#     func=safe_smart_resize,
#     description="等比例图像缩放。只需输入宽或高中的一个，程序会自动计算另一个维度，保证图像不拉伸变形。",
#     params_schema={
#         "width": {
#             "type": "Optional[int]",
#             "range": [32, 2048],
#             "description": "目标宽度。若为 None，则根据高度等比计算。"
#         },
#         "height": {
#             "type": "Optional[int]",
#             "range": [32, 2048],
#             "description": "目标高度。若为 None，则根据宽度等比计算。"
#         }
#     }
# )

global_registry.register(
    name="Gaussian_Blur",
    func=safe_gaussian_blur,
    description="对图像应用高斯模糊。适用于去除白噪点，平滑图像。注意：过度使用会导致图像边缘变得模糊，降低清晰度。",
    params_schema={
        "ksize": {
            "type": "int",
            "range": [1, 15],
            "description": "模糊核的大小。1表示不模糊，值越大降噪效果越强，但图片也会越模糊。建议 Optuna 在 [1, 3, 5, 7] 中进行离散搜索。"
        }
    }
)

# 7. 形态学变换注册
global_registry.register(
    name="Morphology_Cleanup",
    func=safe_morphology_transform,
    description="形态学清理。'open'（开运算）用于去除背景细小噪点；'close'（闭运算）用于填充物体内的细微空洞，常用于二值化后的收尾工作。",
    params_schema={
        "op_type": {
            "type": "string",
            "options": ["open", "close", "dilate", "erode"],
            "description": "变换类型。去噪选 'open'，填补选 'close'。"
        },
        "ksize": {
            "type": "int",
            "range": [1, 11],
            "description": "结构元大小。值越大，消除或填补的力度越大。通常取 3 或 5。"
        }
    }
)

# 8. 自适应二值化注册

global_registry.register(
    name="Adaptive_Binarization",
    func=safe_adaptive_threshold,
    description="智能二值化。自动根据局部区域的光照情况决定阈值，非常适合处理拍摄不均匀、带阴影的文档或扫描件。",
    params_schema={
        "block_size": {
            "type": "int",
            "range": [3, 41],
            "description": "局部邻域大小。值越大，考虑的背景范围越广。建议在 [11, 21, 31] 中搜索。"
        },
        "c": {
            "type": "int",
            "range": [-10, 10],
            "description": "从平均值中减去的常数。正值会让结果更“干净”，负值会保留更多细节。"
        }
    }
)

# 9. 中值滤波注册

global_registry.register(
    name="Median_Denoise",
    func=safe_median_blur,
    description="中值去噪。专门对抗‘椒盐噪声’（黑白斑点）。它在去噪的同时能完美保护图像边缘不被模糊，是工业相机预处理的首选。",
    params_schema={
        "ksize": {
            "type": "int",
            "options": [1, 3, 5, 7, 9],
            "description": "滤波核大小。必须为奇数。值越大去噪能力越强，但会丢失细小结构。"
        }
    }
)

# 10. 自动白平衡注册
global_registry.register(
    name="Auto_White_Balance",
    func=safe_color_balance,
    description="灰度世界假设自动白平衡。修复由于环境光源导致的偏色（如昏黄灯光下的泛黄）。使图像中的白色回归真实白色，提升色彩还原度。",
    params_schema={
        "blend_ratio": {
            "type": "float",
            "range": [0.0, 1.0],
            "description": "白平衡的应用强度。如果是不确定的场景，建议从较低的值（如 0.2）开始搜索。"
        }
    }
)

# 11. 引导滤波注册
global_registry.register(
    name="Guided_Filter",
    func=safe_guided_filter,
    description="引导滤波。最先进的保边平滑算法之一，计算速度快且没有双边滤波常见的梯度翻转（光晕）问题。非常适合作为图像锐化前的底图生成。",
    params_schema={
        "radius": {
            "type": "int",
            "range": [2, 20],
            "description": "滤波半径。值越大，图像越平滑。默认 8。"
        },
        "eps": {
            "type": "float",
            "range": [0.001, 0.1],
            "description": "惩罚项（正则化参数）。值越大，保边效果越弱，图像越接近普通模糊。"
        }
    }
)

# global_registry.register(
#     name="Dehaze_DCP",
#     func=safe_dehaze,
#     description="基于暗通道先验（DCP）的通用去雾算法。专门修复户外由于雾、霾或水汽导致的对比度下降。通过模拟物理大气散射模型，还原图像的真实色彩与清晰度，特别适合监控、航拍及车载视觉预处理。",
#     params_schema={
#         "window_size": {
#             "type": "int",
#             "options": [7, 11, 15, 21, 31],
#             "description": "暗通道搜索窗口。通常取 15。对于高分辨率图像可适当增大。值越大去雾效果越均衡，但边缘处理负担会增加。"
#         },
#         "omega": {
#             "type": "float",
#             "range": [0.7, 0.99],
#             "default": 0.95,
#             "description": "去雾强度因子。取 1.0 为全去雾。建议保留少量（如 0.95），能使远景看起来更有自然深度感，避免画面生硬。"
#         },
#         "t0": {
#             "type": "float",
#             "range": [0.05, 0.2],
#             "default": 0.1,
#             "description": "最小透射率门限。用于保护极浓雾区域不产生严重的像素颗粒噪声。值越大画面越柔和。"
#         },
#         "guided_radius": {
#             "type": "int",
#             "range": [10, 100],
#             "default": 40,
#             "description": "导向滤波半径。关键参数，用于修复透射率图。值越大，物体边缘越不容易产生‘白边’或光晕效果。"
#         },
#         "eps": {
#             "type": "float",
#             "options": [0.01, 0.001, 0.0001],
#             "description": "导向滤波平滑因子。控制精修后的透射率图对原图边缘的贴合程度。"
#         }
#     }
# )

global_registry.register(
    name="Image_Deringing",
    func=safe_deringing,
    description="自适应去环滤镜。专门用于消除 JPEG 压缩或过度锐化产生的边缘‘振铃’（光晕）伪影。该函数通过动态边缘掩码技术，只在伪影区域进行平滑，避免破坏图像中心纹理。",
    params_schema={
        "threshold": {
            "type": "int",
            "range": [10, 100],
            "default": 30,
            "description": "边缘敏感度。值越小，检测到的伪影区域越广；值越大，仅处理最强边缘周围。"
        },
        "blur_sigma": {
            "type": "int",
            "range": [5, 50],
            "default": 20,
            "description": "去环强度（颜色空间标准差）。控制平滑的力度，数值越大对环影的抑制越强。"
        },
        "window_size": {
            "type": "int",
            "options": [5, 7, 9, 11, 15],
            "default": 9,
            "description": "处理半径。对于大尺寸图像或明显的宽光晕，需调大此参数。"
        }
    }
)

# 12. Retinex 增强注册

# global_registry.register(
#     name="Low_Light_Retinex",
#     func=safe_low_light_retinex,
#     description="Retinex 图像增强。模拟人类视觉系统对光照的感知，能大幅提升极端暗光下的图像质量，还原物体的本来颜色和细节。注意该算子速度非常慢，谨慎使用！",
#     params_schema={
#         "sigma_list": {
#             "type": "list",
#             "description": "高斯模糊标准差列表。决定了提取光照背景的尺度，默认 [15, 80, 250] 覆盖高中低频。"
#         }
#     }
# )

# 13. 饱和度增强注册
# global_registry.register(
#     name="Saturation_Boost",
#     func=safe_hsv_saturation,
#     description="HSV线性色彩饱和度调整。在不影响亮度的前提下，让图像中的色彩更鲜艳或更素雅。常用于提升视觉美感。注意该函数为线性调整，倍数过大容易造成色彩溢出！",
#     params_schema={
#         "saturation_scale": {
#             "type": "float",
#             "range": [0.0, 3.0],
#             "description": "缩放倍数。1.0 为原图，>1 变鲜艳，<1 变灰暗（0 为纯黑白）。"
#         }
#     }
# )

global_registry.register(
    name="Saturation_Boost_Nonlinear",
    func=safe_hsv_saturation_nonlinear,
    description="HSV非线性色彩饱和度调整。在不影响亮度的前提下，让图像中的色彩更鲜艳或更素雅。常用于提升视觉美感。注意该函数为非线性调整。",
    params_schema={
        "saturation_scale": {
            "type": "float",
            "range": [0.0, 5.0],
            "description": "缩放倍数。1.0 为原图，>1 变鲜艳，<1 变灰暗（0 为纯黑白）。"
        }
    }
)

global_registry.register(
    name="Vibrance",
    func=safe_vibrance,
    description="基于 LUT 与 Tanh 曲线的自然饱和度调整。通过平滑的非线性映射提升低饱和度区域，天然具备柔性截断（Soft-Clip）特性，完美保护高饱和度区域不发生色彩溢出。",
    params_schema={
        "level": {
            "type": "float",
            "range": [0.0, 3.0],
            "description": "增强强度。无限趋近于 0 时为原图，1.0 为中等自然增益，2.0 为强力增益，3.0 为极限浓艳。"
        }
    }
)

global_registry.register(
    name="Color_Temperature_Tune",
    func=safe_color_temperature,
    description="照片级色温与色调微调。当需要让画面变得更温暖（如黄昏夕阳）或更清冷（如赛博朋克夜景），或者轻微修正偏色时使用。它比自动白平衡更受控，需要谨慎考虑参数范围。",
    params_schema={
        "temperature": {
            "type": "float",
            "range": [-30.0, 30.0],
            "description": "色温值。正数变暖（黄/红），负数变冷（蓝）。0为不调整。"
        },
        "tint": {
            "type": "float",
            "range": [-20.0, 20.0],
            "description": "色调值。正数偏洋红（紫），负数偏绿。通常用于修正荧光灯下的偏绿现象。"
        }
    }
)

# 15. 全局色相平移
global_registry.register(
    name="Global_Hue_Shift",
    func=safe_hue_shift,
    description="全局色相平移。当需要进行彻底的艺术化色彩偏移，或用户明确提出要改变某种颜色时使用。请谨慎调用，因为它会改变画面中所有物体的本来颜色。",
    params_schema={
        "hue_shift": {
            "type": "int",
            "range": [-30, 30],
            "description": "色相旋转角度 (OpenCV 映射域)。0 为原图，轻微调色建议在 [-15, 15] 之间搜索。"
        }
    }
)

# global_registry.register(
#     name="Zero_DCE_Enhance",
#     func=safe_zero_dce,
#     description=(
#         "基于深度学习的轻量级低照度增强算法。不同于传统的直方图均衡化，"
#         "它通过预测非线性增强曲线来提升图像亮度，能够有效避免过曝并抑制暗部噪声。"
#         "特别适用于夜间拍摄或逆光场景的图像修复。"
#         "只支持彩色图像。"
#     ),
#     params_schema={}
# )

global_registry.register(
    name="NL_Means_Denoising",
    func=safe_nl_means_denoise,
    description="非局部均值降噪算子。通过寻找图像内部相似的纹理块进行加权平均，在强力去除高斯白噪声的同时，极其优秀地保留了物体的边缘和纹理细节。效果显著优于普通模糊，但计算耗时相对较高。",
    params_schema={
        "h": {
            "type": "float",
            "range": [3.0, 30.0],
            "description": "滤波强度。值越大降噪越强，但也可能丢失部分细节。对于轻微噪点设为 5-10，严重噪点设为 10-20。默认 10.0。"
        },
        "template_window": {
            "type": "int",
            "options": [5, 7, 9, 11, 13, 15],
            "description": "模板窗口大小(像素)。用于计算权重的像素块尺寸，必须是奇数。一般保持默认 7 即可兼顾效果和性能。"
        },
        "search_window": {
            "type": "int",
            "range": [11, 31],
            "description": "搜索窗口大小(像素)。在多大的范围内寻找相似的像素块，必须是奇数。值越大降噪效果可能越好，但计算耗时会呈平方级增加。默认 21。"
        }
    }
)

if hasattr(cv2, "xphoto") and hasattr(cv2, "ximgproc"):

    # 2. 灰度世界白平衡注册
    # global_registry.register(
    #     name="Grayworld_WB",
    #     func=safe_enhance_grayworld_wb,
    #     description="自动色彩校正。基于灰度世界假设，极其有效地去除图像的全局偏色现象（如偏黄、偏蓝），使色彩还原真实自然。",
    #     params_schema={}  # 无需超参数
    # )

    # 3. Sauvola 自适应二值化注册
    global_registry.register(
        name="Sauvola_Binarization",
        func=safe_enhance_sauvola,
        description="高级局部二值化。专为光照极度不均匀、带有阴影或强光斑的文档和发票图像设计，能够干净地提取前景文字和线条。",
        params_schema={
            "block_size": {
                "type": "int",
                "range": [3, 51],
                "description": "局部计算窗口大小，必须为奇数。值越大对大面积阴影的抵抗力越强，通常取 15 或更大。"
            },
            "k": {
                "type": "float",
                "range": [0.01, 0.5],
                "description": "控制阈值灵敏度。较低的值会保留更多前景（但可能引入噪声），较高的值会过滤更多背景。"
            }
        }
    )

    # 4. 各向异性扩散注册
    global_registry.register(
        name="Anisotropic_Diffusion",
        func=safe_denoise_anisotropic,
        description="智能降噪平滑。基于热力学方程，能在抹平严重杂色噪声（如高强度压缩噪点）的同时，保持甚至锐化物体边缘。",
        params_schema={
            "alpha": {
                "type": "float",
                "range": [0.05, 0.25],
                "description": "扩散系数。控制每次迭代的平滑步长，不可过大以防数值不稳定。"
            },
            "k": {
                "type": "float",
                "range": [5.0, 30.0],
                "description": "边缘敏感度。值越小，越倾向于保留微弱边缘；值越大，越容易平滑掉弱边缘。"
            },
            "niters": {
                "type": "int",
                "range": [3, 20],
                "description": "迭代次数。次数越多平滑越彻底，但耗时也越长。"
            }
        }
    )

    # 5. HDR细节增强注册
    global_registry.register(
        name="HDR_Detail_Enhancement",
        func=safe_enhance_detail,
        description="HDR级别细节增强。极大地提亮暗部并挖掘图像的潜在细节，适合风景、逆光、欠曝场景的提亮和清晰度拉升。",
        params_schema={
            "sigma_s": {
                "type": "float",
                "range": [10.0, 200.0],
                "description": "空间平滑参数。控制细节提取的区域范围，值越大整体对比度影响范围越广。"
            },
            "sigma_r": {
                "type": "float",
                "range": [0.1, 1.0],
                "description": "色彩/范围平滑参数。控制什么程度的色差会被认定为边缘并被增强。"
            }
        }
    )

    # 6. L0梯度平滑注册
    global_registry.register(
        name="L0_Smooth",
        func=safe_smooth_l0,
        description="二次元/色块化平滑。抹除低振幅的杂乱噪点和纹理，产生类似插画、卡通化或纯净色块的效果，绝对保留强边缘。",
        params_schema={
            "kappa": {
                "type": "float",
                "range": [1.5, 3.0],
                "description": "控制梯度截断的锐利度，通常保持在 2.0 左右即可。"
            },
            "lambda_param": {
                "type": "float",
                "range": [0.01, 0.05],
                "description": "平滑权重。值越大，抹除的微小纹理越多，色块化越明显。"
            }
        }
    )

    # 7. 滚动导向滤波注册
    global_registry.register(
        name="Rolling_Guidance_Filter",
        func=safe_filter_rolling_guidance,
        description="结构与纹理分离。用于在严格保留物体整体宏观结构的前提下，彻底抹除表面的复杂纹理（如墙面裂纹、皮肤粗糙感）。",
        params_schema={
            "sigma_color": {
                "type": "float",
                "range": [10.0, 50.0],
                "description": "色彩空间标准差。决定了多少色差会被平滑掉。"
            },
            "sigma_space": {
                "type": "float",
                "range": [2.0, 10.0],
                "description": "坐标空间标准差。决定了多大范围内的纹理会被揉捏在一起抹除。"
            },
            "num_iters": {
                "type": "int",
                "range": [2, 6],
                "description": "滚动迭代次数。次数越多，纹理被剥离得越干净。"
            }
        }
    )

if importlib.util.find_spec("pywt"):
    global_registry.register(
        name="Wavelet_Denoising",
        func=safe_wavelet_denoise,
        description="小波变换频域降噪算子。擅长处理图像中的数字颗粒噪点（如高 ISO 噪点或压缩伪影），能在频域分离噪声并将其抑制。",
        params_schema={
            "levels": {
                "type": "int",
                "range": [1, 5],
                "description": "小波分解层数。层数越高，能去处的低频噪点越多，但耗时增加且可能导致失真。默认 3。"
            }
        }
    )

# global_registry.register(
#     name="Total_Variation_Denoising_Chambolle",
#     func=safe_tv_denoise_chambolle,
#     description="基于 Chambolle 的全变分保边降噪算子。能够有效去除图像中的高频噪点，同时最大程度地保留物体的清晰边缘。适合处理平滑区域多但需要锐利边缘的图像。",
#     params_schema={
#         "weight": {
#             "type": "float",
#             "range": [0.05, 0.5],
#             "description": "去噪权重。值越高，图像越平滑（降噪效果越强），但也可能丢失细节。默认 0.1。"
#         }
#     }
# )

global_registry.register(
    name="Total_Variation_Denoising_Bregman",
    func=safe_tv_denoise_bregman,
    description="基于 Split-Bregman 优化的全变分降噪算子。处理速度极快，适合大分辨率图像。能有效去除噪点并保持边缘锐利，产生类似'卡通化'的平滑效果，适合处理平滑区域多但需要锐利边缘的图像。",
    params_schema={
        "weight": {
            "type": "float",
            "range": [0.5, 20.0],
            "description": "保真度权重（注意：值越小，去噪效果越强！）。较小的值(如 1.0)会大幅抹平图像，较大的值(如 10.0)则更多保留原图细节。默认 5.0。"
        },
        "isotropic": {
            "type": "bool",
            "options": [True, False],
            "description": "是否使用各向同性 TV 降噪。True 通常能更好地处理自然图像的圆滑边缘，False 则倾向于生成横平竖直的块状边缘。默认 True。"
        }
    }
)

global_registry.register(
    name="Adjust_Sigmoid",
    func=safe_adjust_sigmoid,
    description="S曲线对比度增强算子。通过 Sigmoid 函数非线性增强图像中段灰阶的对比度，同时保留亮部和暗部细节，通常会让画面具有通透的'电影感'。",
    params_schema={
        "cutoff": {
            "type": "float",
            "range": [0.0, 1.0],
            "description": "阈值截断点。代表被增强对比度的中间亮度值。低于此值变暗，高于此值变亮。默认 0.5 (即中度灰)。"
        },
        "gain": {
            "type": "float",
            "range": [2.0, 20.0],
            "description": "增益系数。决定了对比度增强的剧烈程度，值越高，中间调对比度越强。默认 10.0。"
        }
    }
)

global_registry.register(
    name="Richardson_Lucy_Deblur",
    func=safe_richardson_lucy,
    description="RL 去模糊算子。基于迭代的去卷积算法，用于恢复失焦或轻微模糊的图像，能够有效找回丢失的高频细节。",
    params_schema={
        "iterations": {
            "type": "int",
            "range": [5, 30],
            "description": "迭代次数。次数越多细节越清晰，但也会指数级放大图像中的噪点并增加计算时间。默认 15。"
        },
        "psf_size": {
            "type": "int",
            "options": [3, 5, 7, 9, 11],
            "description": "模糊核大小(必须为奇数)。假设图像最初受到多大范围的模糊，值越大去模糊范围越广。默认 5。"
        }
    }
)

# global_registry.register(
#     name="Unsupervised_Wiener_Deblur",
#     func=safe_unsupervised_wiener,
#     description="无监督维纳去模糊算子。能在尝试去除图像失焦模糊的同时，自动估算并抑制噪声，结果通常比纯锐化更柔和自然。",
#     params_schema={
#         "psf_size": {
#             "type": "int",
#             "options": [3, 5, 7, 9, 11, 13, 15],
#             "description": "模糊核大小(必须为奇数)。用于估测图像本身的模糊半径。值越大修复的模糊越严重，但也可能带来光晕效应。默认 5。"
#         }
#     }
# )

global_registry.load_custom_tools()

__all__ = ["global_registry"] # 向全局暴露该注册