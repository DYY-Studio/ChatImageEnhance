import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np 
import base64
import httpx

from openai import OpenAI, DefaultHttpxClient
from typing import BinaryIO, Literal, Sequence

def get_executable_dir():
    if hasattr(sys, 'frozen'):
        return Path(sys.executable).parent.resolve()
    else:
        return Path(__file__).parent.resolve()
    
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
    mode: Literal["binary", "base64", "array"],
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
    if mode in ("binary", "base64", ):
        succ, enc_img = cv2.imencode(img_format, resized_array, img_enc_params)
        if succ:
            if mode == "binary":
                return enc_img.tobytes()
            elif mode == "base64":
                return f"data:image/{'jpeg' if img_format == '.jpg' else 'png'};base64,{base64.b64encode(enc_img.tobytes()).decode()}"
        else:
            return None
    elif mode == "array":
        return resized_array
    
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
                return OpenAI(base_url=base_url, api_key=api_key, max_retries=0, http_client=client, timeout=20.0)
            except Exception as e:
               print(e)
        return OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=20.0)
    except Exception as e:
        print(e)
        return None

def clear_models():
    st.session_state.models = None

def get_models():
    if not st.session_state.api_url or (st.session_state.has_api_key and not st.session_state.api_key):
        return
    try:
        models = get_openai_client(st.session_state.api_url, st.session_state.api_key, st.session_state.proxy_url).models.list()
        if models:
            st.session_state.models = [model.id for model in models]
        else:
            st.session_state.models = None
    except Exception as e:
        print(e)
        pass