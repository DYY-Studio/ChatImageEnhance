import torch
import os
from PIL import Image
import numpy as np
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor
from deeplearning.prepare import CFG, Evaluator
import torch.nn.functional as F
from PIL import ImageOps
# 初始化设备
device = torch.device(CFG["device"])

# 加载模型和处理器
model_id = "caidas/swin2SR-compressed-sr-x4-48"
print(f"Loading model {model_id}...")
processor = AutoImageProcessor.from_pretrained(model_id)
model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
print("Model loaded successfully.")

# 初始化评估器
evaluator = Evaluator()

# 遍历输入目录处理图像
input_dir = CFG["input_dir"]
output_dir = CFG["output_dir"]
gt_dir = CFG["gt_dir"]

for filename in os.listdir(input_dir):
    # 1. 读取图像
    input_path = os.path.join(input_dir, filename)
    image = Image.open(input_path).convert("RGB")
    w, h = image.size

    # 2. 对 PIL 图像进行 Padding (补齐到 64 的倍数)
    # 使用 ImageOps.expand 在右侧和下方填充，颜色设为 0 (黑色) 或边缘反射
    pad_w = (64 - w % 64) % 64
    pad_h = (64 - h % 64) % 64
    # ImageOps.expand 参数顺序: (left, top, right, bottom)
    padded_image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0)

    # 3. 推理准备
    # 使用 processor 转换为 tensor，确保不做额外的 resize
    inputs = processor(images=padded_image, return_tensors="pt").to(device)

    # 4. 执行推理
    with torch.no_grad():
        outputs = model(**inputs)
        # 获取重建图像并转回 CPU
        output = outputs.reconstruction.squeeze(0).cpu().clamp(0, 1)

        # 5. 后处理与精准剪裁
        # 将 Tensor 转回 PIL
        output_img_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        full_output_image = Image.fromarray(output_img_np)

        # 【关键修改】：模型是 x4 的，所以这里必须乘以 4
        scale_factor = 4
        final_w, final_h = w * scale_factor, h * scale_factor

        # 现在裁剪出的就是完整的超分图像
        output_image = full_output_image.crop((0, 0, final_w, final_h))

        # 6. 保存与评估
        output_path = os.path.join(output_dir, filename)
        output_image.save(output_path)

        gt_path = os.path.join(gt_dir, filename)
        if os.path.exists(gt_path):
            gt_image = Image.open(gt_path).convert("RGB")
            # 【关键修改】：GT 也要缩放到 x4 尺寸进行像素级对比
            if gt_image.size != (final_w, final_h):
                gt_image = gt_image.resize((final_w, final_h), Image.LANCZOS)

            p, s, ms = evaluator.metrics(output_image, gt_image)
            print(f"[{filename}] x4 Metrics: PSNR={p:.2f}, SSIM={s:.4f}")

print("\nProcessing completed.")


def main():
    """主函数，用于外部调用图像增强功能"""
    # 初始化设备
    device = torch.device(CFG["device"])
    
    # 加载模型和处理器
    model_id = "caidas/swin2SR-compressed-sr-x4-48"
    print(f"Loading model {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
    print("Model loaded successfully.")
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 遍历输入目录处理图像
    input_dir = CFG["input_dir"]
    output_dir = CFG["output_dir"]
    gt_dir = CFG["gt_dir"]
    
    processed_images = []
    
    for filename in os.listdir(input_dir):
        try:
            # 1. 读取图像
            input_path = os.path.join(input_dir, filename)
            if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
                continue  # 跳过非图像文件
                
            image = Image.open(input_path).convert("RGB")
            w, h = image.size
    
            # 2. 对 PIL 图像进行 Padding (补齐到 64 的倍数)
            pad_w = (64 - w % 64) % 64
            pad_h = (64 - h % 64) % 64
            padded_image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0)
    
            # 3. 推理准备
            inputs = processor(images=padded_image, return_tensors="pt").to(device)
    
            # 4. 执行推理
            with torch.no_grad():
                outputs = model(**inputs)
                # 获取重建图像并转回 CPU
                output = outputs.reconstruction.squeeze(0).cpu().clamp(0, 1)
    
                # 5. 后处理与精准剪裁
                output_img_np = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                full_output_image = Image.fromarray(output_img_np)
    
                # 【关键修改】：模型是 x4 的，所以这里必须乘以 4
                scale_factor = 4
                final_w, final_h = w * scale_factor, h * scale_factor
    
                # 现在裁剪出的就是完整的超分图像
                output_image = full_output_image.crop((0, 0, final_w, final_h))
    
                # 6. 保存与评估
                output_path = os.path.join(output_dir, filename)
                output_image.save(output_path)
                
                processed_images.append({
                    'input_path': input_path,
                    'output_path': output_path,
                    'original_size': (w, h),
                    'enhanced_size': (final_w, final_h)
                })
    
                gt_path = os.path.join(gt_dir, filename)
                if os.path.exists(gt_path):
                    gt_image = Image.open(gt_path).convert("RGB")
                    # 【关键修改】：GT 也要缩放到 x4 尺寸进行像素级对比
                    if gt_image.size != (final_w, final_h):
                        gt_image = gt_image.resize((final_w, final_h), Image.LANCZOS)
    
                    p, s, ms = evaluator.metrics(output_image, gt_image)
                    print(f"[{filename}] x4 Metrics: PSNR={p:.2f}, SSIM={s:.4f}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    print(f"\n共处理了 {len(processed_images)} 张图片")
    return processed_images


if __name__ == "__main__":
    main()