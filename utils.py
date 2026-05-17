import sys
from pathlib import Path
import platform
import subprocess

import streamlit as st
import cv2
import numpy as np 
import base64
import httpx
import torch

from openai import OpenAI, DefaultHttpxClient
from typing import BinaryIO, Literal, Sequence

@st.cache_resource
def get_openai_client(base_url: str, api_key: str, proxy_url: str):
    try:
        if proxy_url:
            try:
                client = DefaultHttpxClient(
                    transport=httpx.HTTPTransport(
                        proxy=proxy_url
                    )
                )
                return OpenAI(base_url=base_url, api_key="dummykey" if api_key is None or not api_key else api_key, max_retries=0, http_client=client, timeout=20.0)
            except Exception as e:
               print(e)
        return OpenAI(base_url=base_url, api_key="dummykey" if api_key is None or not api_key else api_key, max_retries=0, timeout=20.0)
    except Exception as e:
        print(e)
        return None

@st.cache_resource
def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
    if hasattr(torch, "npu") and torch.npu.is_available():
        devices.append("npu")
    devices.append("cpu")
    return devices

def _run_subprocess(args: list[str], timeout: int = 6) -> str:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False
        )
        if proc.returncode == 0:
            return (proc.stdout or "").strip()
        return (proc.stderr or "").strip()
    except Exception:
        return ""

@st.cache_resource
def get_device_info_subprocess() -> dict:
    """
    通过 subprocess 获取跨平台设备信息（优先 GPU）。
    """
    system = platform.system().lower()
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": [],
        "cpu": "",
        "raw": {}
    }

    # 1) NVIDIA（跨平台最稳定）
    nvidia = _run_subprocess([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader"
    ])
    if nvidia and all(
        err not in nvidia.lower()
        for err in ("not found", "not recognized", "无法将", "command not found")
    ):
        lines = [line.strip() for line in nvidia.splitlines() if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["gpu"].append({
                    "vendor": "NVIDIA",
                    "name": parts[0],
                    "memory_total": parts[1],
                    "driver": parts[2]
                })
        info["raw"]["nvidia_smi"] = nvidia

    # 2) AMD ROCm（Linux 常见）
    if not info["gpu"]:
        rocm = _run_subprocess(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
        if rocm and "not found" not in rocm.lower():
            info["gpu"].append({
                "vendor": "AMD/ROCm",
                "name": "Detected by rocm-smi",
                "memory_total": "See raw",
                "driver": ""
            })
            info["raw"]["rocm_smi"] = rocm

    # 3) macOS 显卡信息
    if system == "darwin" and not info["gpu"]:
        sp = _run_subprocess(["system_profiler", "SPDisplaysDataType"])
        if sp:
            info["gpu"].append({
                "vendor": "Apple",
                "name": "Detected by system_profiler",
                "memory_total": "See raw",
                "driver": ""
            })
            info["raw"]["system_profiler"] = sp

    # 4) CPU 信息
    if system == "windows":
        cpu = _run_subprocess(["wmic", "cpu", "get", "Name"])
        if cpu:
            cpu_lines = [line.strip() for line in cpu.splitlines() if line.strip() and line.strip().lower() != "name"]
            info["cpu"] = cpu_lines[0] if cpu_lines else platform.processor()
    elif system == "linux":
        cpu = _run_subprocess(["lscpu"])
        if cpu:
            info["raw"]["lscpu"] = cpu
            model_line = next((ln for ln in cpu.splitlines() if "Model name:" in ln), "")
            info["cpu"] = model_line.split(":", 1)[1].strip() if ":" in model_line else platform.processor()
    elif system == "darwin":
        cpu = _run_subprocess(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["cpu"] = cpu or platform.processor()
    else:
        info["cpu"] = platform.processor()

    if not info["cpu"]:
        info["cpu"] = platform.processor()

    return info

def format_device_info_for_prompt(device_info: dict) -> str:
    if not isinstance(device_info, dict):
        return "设备信息不可用"
    lines = [
        f"平台: {device_info.get('platform', 'unknown')}",
        f"Python: {device_info.get('python', 'unknown')}",
        f"CPU: {device_info.get('cpu', 'unknown')}"
    ]
    gpus = device_info.get("gpu") or []
    if gpus:
        for idx, gpu in enumerate(gpus):
            lines.append(
                f"GPU[{idx}]: {gpu.get('vendor','')} {gpu.get('name','')} | VRAM={gpu.get('memory_total','?')} | Driver={gpu.get('driver','')}"
            )
    else:
        lines.append("GPU: 未通过系统命令检测到可用设备")
    return "\n".join(lines)

@st.cache_resource()
def get_executable_dir():
    if hasattr(sys, 'frozen'):
        return Path(sys.executable).parent.resolve()
    else:
        return Path(__file__).parent.resolve()
    
@st.cache_resource()
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).parent.resolve()
    
    return str(base_path / relative_path)

@st.cache_data
def load_bgr_img_from_file(file: BinaryIO) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

@st.cache_data
def get_encoded_img(raw_array: np.ndarray) -> bytes:
    succ, enc_img = cv2.imencode('.png', raw_array, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    return succ, enc_img.tobytes()

def get_thumbnail_size(raw_array: np.ndarray, max_side: int = 800) -> tuple[bool, tuple[int, int]]:
    h, w = raw_array.shape[:2]
    current_max = max(h, w)
    if current_max > max_side:
        scale = max_side / current_max
        new_size = (int(w * scale), int(h * scale), )
        new_min = min(new_size)
        if new_min < 25:
            scale = scale * (25 / new_min)
        return True, (int(w * scale), int(h * scale))
    return False, (w, h)

@st.cache_data
def get_thumbnail_img(
    raw_array: np.ndarray, 
    mode: Literal["binary", "b64", "array"],
    max_side: int = 800, 
    interpolation: int = cv2.INTER_AREA,
    img_format: Literal[".jpg", ".png"] = ".jpg",
    img_enc_params: Sequence[int] = [cv2.IMWRITE_JPEG_QUALITY, 95]
) -> bytes | str | np.ndarray | None:
    do_resize, new_size = get_thumbnail_size(raw_array, max_side)
    if do_resize:
        resized_array = cv2.resize(raw_array, new_size, interpolation=interpolation)
    else:
        resized_array = raw_array
    if mode in ("binary", "b64", ):
        succ, enc_img = cv2.imencode(img_format, resized_array, img_enc_params)
        if succ:
            if mode == "binary":
                return enc_img.tobytes()
            elif mode == "b64":
                return f"data:image/{'jpeg' if img_format == '.jpg' else 'png'};base64,{base64.b64encode(enc_img.tobytes()).decode()}"
        else:
            return None
    elif mode == "array":
        return resized_array

def get_thumbnail_img_nocache(
    raw_array: np.ndarray, 
    mode: Literal["binary", "b64", "array"],
    max_side: int = 800, 
    interpolation: int = cv2.INTER_AREA,
    img_format: Literal[".jpg", ".png"] = ".jpg",
    img_enc_params: Sequence[int] = [cv2.IMWRITE_JPEG_QUALITY, 95]
) -> bytes | str | np.ndarray | None:
    do_resize, new_size = get_thumbnail_size(raw_array, max_side)
    if do_resize:
        resized_array = cv2.resize(raw_array, new_size, interpolation=interpolation)
    else:
        resized_array = raw_array
    if mode in ("binary", "b64", ):
        succ, enc_img = cv2.imencode(img_format, resized_array, img_enc_params)
        if succ:
            if mode == "binary":
                return enc_img.tobytes()
            elif mode == "b64":
                return f"data:image/{'jpeg' if img_format == '.jpg' else 'png'};base64,{base64.b64encode(enc_img.tobytes()).decode()}"
        else:
            return None
    elif mode == "array":
        return resized_array

def clear_models():
    st.session_state.models = None
    get_openai_client.clear()

def get_models():
    if not st.session_state.api_url or (st.session_state.has_api_key and not st.session_state.api_key):
        return
    try:
        models = get_openai_client(
            st.session_state.api_url, 
            st.session_state.api_key if st.session_state.has_api_key else "dummykey", 
            st.session_state.proxy_url
        ).models.list()
        if models:
            st.session_state.models = [model.id for model in models]
        else:
            st.session_state.models = None
    except Exception as e:
        print(e)
        pass
