from utils import get_executable_dir

import numpy as np
import torch
import cv2

from models.ZeroDCE import ZeroDCE_Extension_NET
def safe_zero_dce(img: np.ndarray, cache: dict | None = None, device: str = 'cpu'):
    calc_size: int = 512
    try:
        h_high, w_high = img.shape[:2]
        scale = min(calc_size / h_high, calc_size / w_high)
        
        if scale < 1.0:
            new_w, new_h = int(w_high * scale), int(h_high * scale)
            img_lowres = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_lowres = img.copy()

        img_lowres_tensor = torch.from_numpy(img_lowres / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        DCE_net = cache.get("zero_dce_model") if cache is not None else None
        if DCE_net is None:
            DCE_net = ZeroDCE_Extension_NET().to(device)
            state_dict = torch.load(
                str(get_executable_dir() / "models/ZeroDCE_Epoch99.pth"),
                map_location=torch.device(device),
                weights_only=True
            )
            DCE_net.load_state_dict(state_dict)
            DCE_net.eval()
            if cache is not None:
                cache["zero_dce_model"] = DCE_net

        with torch.inference_mode():
            param_tensor = DCE_net(img_lowres_tensor)

        params_lowres = param_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
       
        if scale < 1.0:
            params_highres = cv2.resize(params_lowres, (w_high, h_high), interpolation=cv2.INTER_LINEAR)
        else:
            params_highres = params_lowres
    
        y = img.astype(np.float32) / 255.0
        
        for _ in range(8):
            y = y + params_highres * (np.square(y) - y)
            
        result_array = np.clip(y * 255.0, 0, 255).astype(np.uint8)
        return result_array
    except Exception as e:
        raise RuntimeError(f"ZeroDCE failed: {str(e)}")
    
from models.SCI import SCIRuntime
def safe_sci_enhance(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    try:
        input_tensor = torch.from_numpy(img / 255.0).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        sci_model = cache.get("sci_runtime_model") if cache is not None else None
        if sci_model is None:
            sci_model = SCIRuntime().to(device)
            sci_model.eval()
            
            # 安全的权重加载逻辑 (兼容原作者包含了大量冗余层的字典)
            state_dict = torch.load(
                str(get_executable_dir() / "models/SCI_weights_1_3500.pt"),
                map_location=device, 
                weights_only=True
            )
            model_dict = sci_model.state_dict()
            # 过滤掉不需要的权重 (如 hb, calibrate 相关的权重)
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            sci_model.load_state_dict(model_dict)
            
            if cache is not None: cache["sci_runtime_model"] = sci_model

        with torch.inference_mode():
            enhanced_tensor = sci_model(input_tensor)

        result_tensor = enhanced_tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
        result_array = (result_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        return result_array

    except Exception as e:
        raise RuntimeError(f"SCI processing failed: {str(e)}")

    finally:
        if 'input_tensor' in locals(): del input_tensor
        if 'enhanced_tensor' in locals(): del enhanced_tensor

from models.FFDNet import FFDNet
def safe_ffdnet_denoise(
    img: np.ndarray, 
    noise_level: float = 15.0,
    cache: dict | None = None, 
    device: str = 'cpu'
):
    try:
        input_tensor = torch.from_numpy(img / 255.0).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        ffdnet_model = cache.get("ffdnet_model") if cache is not None else None
        if ffdnet_model is None:
            ffdnet_model = FFDNet(
                in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R'
            )
            for k, v in ffdnet_model.named_parameters():
                v.requires_grad = False

            ffdnet_model.to(device)
            ffdnet_model.eval()
            
            state_dict = torch.load(
                str(get_executable_dir() / "models/ffdnet_color_clip.pth"),
                map_location=device, 
                weights_only=True
            )
            ffdnet_model.load_state_dict(state_dict)
            
            if cache is not None: cache["ffdnet_model"] = ffdnet_model

        sigma_tensor = torch.full((1, 1, 1, 1), noise_level / 255.0).type_as(input_tensor)

        with torch.inference_mode():
            enhanced_tensor = ffdnet_model(input_tensor, sigma_tensor)

        result_tensor = enhanced_tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
        result_array = (result_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        return result_array

    except Exception as e:
        raise RuntimeError(f"SCI processing failed: {str(e)}")

    finally:
        if 'input_tensor' in locals(): del input_tensor
        if 'enhanced_tensor' in locals(): del enhanced_tensor