from models.ZeroDCE import ZeroDCE_NET
from utils import get_executable_dir

import numpy as np
import torch

def safe_zero_dce(img: np.ndarray, cache: dict | None = None, device: str = 'cpu'):
    try:
        data_lowlight = img / 255.0
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2,0,1)
        data_lowlight = data_lowlight.to(device).unsqueeze(0)

        DCE_net = cache.get("zero_dce_model") if cache is not None else None
        if DCE_net is None:
            DCE_net = ZeroDCE_NET().to(device)
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
            enhanced_image = DCE_net(data_lowlight)

        result_tensor = enhanced_image.detach().cpu().squeeze(0)
        result_tensor = result_tensor.permute(1, 2, 0)

        result_array = (result_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        return result_array
    except Exception as e:
        raise RuntimeError(f"ZeroDCE failed: {str(e)}")