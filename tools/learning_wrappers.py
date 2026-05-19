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

    finally:
        if 'img_lowres_tensor' in locals(): del img_lowres_tensor
        if 'param_tensor' in locals(): del param_tensor
    
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

from models.SepLUT import SepLUTGenerator, apply_lut1d_opencv, apply_lut3d_pil
def safe_seplut_retouch(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    calc_size: int = 512
    try:
        h_high, w_high = img.shape[:2]
        scale = min(calc_size / h_high, calc_size / w_high)
        
        if scale < 1.0:
            new_w, new_h = int(w_high * scale), int(h_high * scale)
            img_lowres = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_lowres = img.copy()

        input_tensor = torch.from_numpy(img_lowres / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        seplut_model = cache.get("seplut_model") if cache is not None else None
        if seplut_model is None:
            seplut_model = SepLUTGenerator(n_base_feats=8, n_vertices_1d=17, n_vertices_3d=17)
            for k, v in seplut_model.named_parameters():
                v.requires_grad = False

            seplut_model.to(device)
            seplut_model.eval()
            
            state_dict = torch.load(
                str(get_executable_dir() / "models/SepLUT-FiveK-sRGB-M8#3D17#1D17.pth"),
                map_location=device, 
                weights_only=True
            )['state_dict']
            seplut_model.load_state_dict(state_dict)
            
            if cache is not None: cache["seplut_model"] = seplut_model

        with torch.inference_mode():
            enhanced_1d_lut, enhanced_3d_lut = seplut_model(input_tensor)

        result = apply_lut3d_pil(apply_lut1d_opencv(img, enhanced_1d_lut), enhanced_3d_lut)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"SepLUT processing failed: {str(e)}")

    finally:
        if 'input_tensor' in locals(): del input_tensor
        if 'img_lowres' in locals(): del img_lowres

from models.ImageAdaptive3DLUT import apply_lut_to_image, load_adaptive_lut_model
def safe_ia3dlut_retouch(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    calc_size: int = 512
    try:
        h_high, w_high = img.shape[:2]
        scale = min(calc_size / h_high, calc_size / w_high)
        
        if scale < 1.0:
            new_w, new_h = int(w_high * scale), int(h_high * scale)
            img_lowres = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_lowres = img.copy()

        input_tensor = torch.from_numpy(img_lowres / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

        model = cache.get("ia3dlut_model") if cache is not None else None
        if model is None:
            model = load_adaptive_lut_model(
                str(get_executable_dir() / "models"),
                device=device
            )

            if cache is not None: cache["ia3dlut_model"] = model

        result_rgb = apply_lut_to_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), input_tensor, model, device)
        result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Image Adaptive processing failed: {str(e)}")
    finally:
        if 'input_tensor' in locals(): del input_tensor
        if 'img_lowres' in locals(): del img_lowres

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from core.model_assets import MODEL_IGNORE_PATTERNS

def _sync_modelscope_pipeline_device(image_pipeline):
    pipeline_device = getattr(image_pipeline, "device", None)
    if pipeline_device is not None and hasattr(image_pipeline, "_device"):
        image_pipeline._device = pipeline_device


def _normalize_modelscope_device(device: str) -> str:
    normalized = str(device or "cpu").strip().lower()
    if normalized in {"cpu", "cuda", "gpu"}:
        return normalized
    if normalized.startswith("cuda:") or normalized.startswith("gpu:"):
        return normalized
    return "cpu"


def _modelscope_output_to_bgr(output_img: np.ndarray, output_color_space: str) -> np.ndarray:
    result = np.asarray(output_img)
    color_space = str(output_color_space or "bgr").strip().lower()

    if color_space == "bgr":
        return result.copy()
    if color_space == "rgb":
        if result.ndim == 3 and result.shape[2] == 4:
            return cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
        if result.ndim == 3 and result.shape[2] == 3:
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result.copy()

    raise ValueError(f"Unsupported ModelScope output color space: {output_color_space}")


def _modelscope_img_pipeline(
    img: np.ndarray,
    model_name: str,
    cache_key: str,
    pipeline_task: str,
    caller_name: str,
    cache: dict | None = None,
    device: str = 'cpu',
    output_color_space: str = 'bgr'
):
    try:
        modelscope_device = _normalize_modelscope_device(device)
        image_pipeline = cache.get(cache_key, None) if cache is not None else None
        cached_device = str(getattr(image_pipeline, "device_name", "")).lower()
        if image_pipeline is not None and cached_device != modelscope_device:
            # 设备变更: 显式从缓存中移除旧 pipeline 以释放其占用的显存
            if cache is not None:
                cache.pop(cache_key, None)
            del image_pipeline
            image_pipeline = None

        if image_pipeline is None:
            try:
                image_pipeline = pipeline(
                    pipeline_task, 
                    model_name,
                    device=modelscope_device,
                    ignore_file_pattern=list(MODEL_IGNORE_PATTERNS)
                )
            except Exception:
                image_pipeline = pipeline(
                    pipeline_task, 
                    model_name,
                    device='cpu',
                    ignore_file_pattern=list(MODEL_IGNORE_PATTERNS)
                )
        if cache is not None:
            cache[cache_key] = image_pipeline
        _sync_modelscope_pipeline_device(image_pipeline)

        img_bgr = np.ascontiguousarray(img)
        result_img = image_pipeline(img_bgr)[OutputKeys.OUTPUT_IMG]
        result = _modelscope_output_to_bgr(result_img, output_color_space)
        return result
    except Exception as e:
        raise RuntimeError(f'{caller_name} failed: {str(e)}')
    finally:
        if 'img_bgr' in locals(): del img_bgr
        if 'result_img' in locals(): del result_img

def safe_nafnet_denoise(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    model_name = 'iic/cv_nafnet_image-denoise_sidd'
    cache_key = 'nafnet_denoise_pipeline'
    pipeline_task = Tasks.image_denoising
    return _modelscope_img_pipeline(
        img=img, model_name=model_name, cache_key=cache_key,
        pipeline_task=pipeline_task, caller_name='NAFNet denoise',
        cache=cache, device=device
    )

def safe_nafnet_demotionblur(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    model_name = 'iic/cv_nafnet_image-deblur_gopro'
    cache_key = 'nafnet_deblur_pipeline'
    pipeline_task = Tasks.image_deblurring
    return _modelscope_img_pipeline(
        img=img, model_name=model_name, cache_key=cache_key,
        pipeline_task=pipeline_task, caller_name='NAFNet demotionblur',
        cache=cache, device=device
    )

def safe_nafnet_demotionblur_and_compress(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    model_name = 'iic/cv_nafnet_image-deblur_reds'
    cache_key = 'nafnet_deblur_compress_compressed_pipeline'
    pipeline_task = Tasks.image_deblurring
    return _modelscope_img_pipeline(
        img=img, model_name=model_name, cache_key=cache_key,
        pipeline_task=pipeline_task, caller_name='NAFNet demotionblur and compress',
        cache=cache, device=device
    )

def safe_uhdm_demoireing(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    model_name = 'iic/cv_uhdm_image-demoireing'
    cache_key = 'uhdm_image_demoireing_pipeline'
    pipeline_task = Tasks.image_demoireing
    return _modelscope_img_pipeline(
        img=img, model_name=model_name, cache_key=cache_key,
        pipeline_task=pipeline_task, caller_name='UHDM demoireing',
        cache=cache, device=device, output_color_space='rgb'
    )

def safe_csrnet_color_enhance(
    img: np.ndarray, 
    cache: dict | None = None, 
    device: str = 'cpu'
):
    from modelscope.models import Model
    from modelscope.preprocessors import Preprocessor
    model_name = 'iic/cv_csrnet_image-color-enhance-models'
    cache_key = 'deeplpfnet_image-color-enhance_pipeline'

    try:
        model = cache.get(cache_key) if cache is not None else None
        if model is None:
            model = Model.from_pretrained(model_name).to(device)
            model.eval()
            for k, v in model.named_parameters():
                v.requires_grad = False

            model.to(device)
            model.eval()
            
            if cache is not None: cache[cache_key] = model
        
        preprocesser = cache.get(f"{cache_key}_pre") if cache is not None else None
        if preprocesser is None:
            preprocesser = Preprocessor.from_pretrained(model_name)

            if cache is not None: cache[f"{cache_key}_pre"] = preprocesser

        input_tensor = torch.from_numpy(preprocesser(img) / 255.0).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.inference_mode():
            enhanced_tensor = model._inference_forward(input_tensor)['outputs']

        result_tensor = enhanced_tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
        result_array = (result_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

        return result_array

    except Exception as e:
        raise RuntimeError(f"CSRNet processing failed: {str(e)}")

    finally:
        if 'input_tensor' in locals(): del input_tensor
        if 'enhanced_tensor' in locals(): del enhanced_tensor
