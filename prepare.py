import os
import requests
import rarfile
import cv2
import numpy as np
import json
import torch
# 统一使用 piq 计算所有指标
from piq import psnr as piq_psnr
from piq import ssim as piq_ssim
from piq import multi_scale_ssim as piq_ms_ssim

# ===================== 【路径配置】 =====================
rarfile.UNRAR_TOOL = r".\UnRAR.exe"
rarfile.NO_CRC_CHECK = True
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_cv")
DATA_URL = "http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar"
RAR_PATH = os.path.join(CACHE_DIR, "waterloo_data.rar")
EXTRACT_DIR = os.path.join(CACHE_DIR, "extracted")

# 解压后的原图目录
PRISTINE_DIR = os.path.join(EXTRACT_DIR, "exploration_database_and_code", "pristine_images")
# 生成的任务数据集目录
DISTORTED_ROOT_DIR = os.path.join(CACHE_DIR, "distorted_dataset")

TOTAL_IMAGES = 250
NOISE_SEED = 12345


# ===================== 1. 数据下载与解压 =====================
def download_and_extract():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(RAR_PATH):
        print(f"正在下载数据集: {DATA_URL}")
        response = requests.get(DATA_URL, stream=True)
        with open(RAR_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    if not os.path.exists(EXTRACT_DIR):
        print("正在解压 (请确保系统已安装 UnRAR)...")
        with rarfile.RarFile(RAR_PATH) as rf:
            rf.extractall(EXTRACT_DIR)
        print("解压完成。")


# ===================== 2. 失真生成逻辑 =====================
def distortion_generator(image, dist_type, level):
    # 复刻原版参数
    gblur_level = [7, 15, 39, 91, 199]
    jpeg_level = [43, 12, 7, 4, 0]
    wn_level = [0.01, 0.03, 0.05, 0.08, 0.1]

    if dist_type == 1:  # 高斯模糊
        hsize = gblur_level[level - 1]
        sigma = hsize / 6
        return cv2.GaussianBlur(image, (hsize, hsize), sigma), "Gaussian Blur"

    elif dist_type == 2:  # 高斯白噪声
        np.random.seed(NOISE_SEED)
        img_float = image.astype(np.float32)
        actual_var = wn_level[level - 1] * (255 ** 2)
        sigma = np.sqrt(actual_var)
        noise = np.random.normal(0, sigma, img_float.shape)
        img_noisy = np.clip(img_float + noise, 0, 255).astype(np.uint8)
        return img_noisy, "Gaussian White Noise"

    elif dist_type == 3:  # JPEG 压缩
        quality = jpeg_level[level - 1]
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED), "JPEG Compression"


# ===================== 3. 统一指标计算 (仅使用 piq) =====================
def get_all_metrics(target_img, ref_img):
    """
    仅使用 piq 调用 PSNR, SSIM, MS-SSIM
    """

    # 将 OpenCV BGR 转换为 Torch RGB [0, 1] 张量
    def preprocess(img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)  # 增加 Batch 维度 (1, C, H, W)

    t_tensor = preprocess(target_img)
    r_tensor = preprocess(ref_img)

    # 检查是否有 GPU 可用以加速计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_tensor = t_tensor.to(device)
    r_tensor = r_tensor.to(device)

    # 计算指标
    with torch.no_grad():
        # PSNR
        p_val = piq_psnr(t_tensor, r_tensor, data_range=1.0)
        # SSIM
        s_val = piq_ssim(t_tensor, r_tensor, data_range=1.0)
        # MS-SSIM (Multi-Scale SSIM)
        ms_val = piq_ms_ssim(t_tensor, r_tensor, data_range=1.0)

    return {
        "psnr": round(float(p_val.item()), 4),
        "ssim": round(float(s_val.item()), 4),
        "ms_ssim": round(float(ms_val.item()), 4)
    }


# ===================== 4. 数据准备主循环 =====================
def prepare_dataset():
    os.makedirs(DISTORTED_ROOT_DIR, exist_ok=True)
    metadata = []

    print(f"开始处理 {TOTAL_IMAGES} 张图片...")

    for img_num in range(1, TOTAL_IMAGES + 1):
        img_name = f"{img_num:05d}.bmp"
        src_path = os.path.join(PRISTINE_DIR, img_name)

        if not os.path.exists(src_path):
            continue

        img = cv2.imread(src_path)
        img_folder = os.path.join(DISTORTED_ROOT_DIR, f"task_{img_num:05d}")
        os.makedirs(img_folder, exist_ok=True)

        # 保存原图
        ref_path = os.path.join(img_folder, "ref_gt.bmp")
        cv2.imwrite(ref_path, img)

        # 预设三个典型任务：模糊、噪声、压缩
        tasks_config = [
            (1, 2, "blur"),
            (2, 2, "noise"),
            (3, 2, "jpeg")
        ]

        for dist_type, lv, label in tasks_config:
            dist_img, dist_desc = distortion_generator(img, dist_type, lv)
            dist_path = os.path.join(img_folder, f"input_{label}.png")
            cv2.imwrite(dist_path, dist_img)

            # 记录元数据
            task_info = {
                "img_id": img_num,
                "input_path": dist_path,
                "gt_path": ref_path,
                "prompt": f"This image has {dist_desc}. Use a CV algorithm to restore it.",
                "metrics_before": get_all_metrics(dist_img, img)
            }
            metadata.append(task_info)

        if img_num % 50 == 0:
            print(f"进度: {img_num}/{TOTAL_IMAGES}")

    # 导出元数据
    with open(os.path.join(DISTORTED_ROOT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n准备完成！数据保存在: {DISTORTED_ROOT_DIR}")


if __name__ == "__main__":
    download_and_extract()
    prepare_dataset()