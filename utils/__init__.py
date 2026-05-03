import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np 
import base64

from typing import BinaryIO

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

@st.cache_data
def get_thumbnail_img(raw_array: np.ndarray, max_side: int = 800, interpolation: int = cv2.INTER_LANCZOS4) -> bytes:
    h, w = raw_array.shape[:2]
    current_max = max(h, w)
    if current_max > max_side:
        scale = max_side / current_max
        resized_array = cv2.resize(raw_array, (int(w * scale), int(h * scale)), interpolation=interpolation)
    else:
        resized_array = raw_array
    succ, enc_img = cv2.imencode('.jpg', resized_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if succ:
        return enc_img.tobytes()
    
@st.cache_data
def get_thumbnail_img_base64(raw_array: np.ndarray, max_side: int = 800, interpolation: int = cv2.INTER_LANCZOS4) -> str:
    h, w = raw_array.shape[:2]
    current_max = max(h, w)
    if current_max > max_side:
        scale = max_side / current_max
        resized_array = cv2.resize(raw_array, (int(w * scale), int(h * scale)), interpolation=interpolation)
    else:
        resized_array = raw_array
    succ, enc_img = cv2.imencode('.jpg', resized_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if succ:
        return f"data:image/jpeg;base64,{base64.b64encode(enc_img.tobytes()).decode()}"

@st.cache_data
def get_thumbnail_img_rgb_array(raw_array: np.ndarray, max_side: int = 800, interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    h, w = raw_array.shape[:2]
    current_max = max(h, w)
    if current_max > max_side:
        scale = max_side / current_max
        resized_array = cv2.resize(raw_array, (int(w * scale), int(h * scale)), interpolation=interpolation)
    else:
        resized_array = raw_array
    return cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB) if raw_array.shape[2] == 3 else resized_array

def get_thumbnail_img_nocache(raw_array: np.ndarray, max_side: int = 800, interpolation: int = cv2.INTER_LANCZOS4) -> bytes:
    h, w = raw_array.shape[:2]
    current_max = max(h, w)
    if current_max > max_side:
        scale = max_side / current_max
        resized_array = cv2.resize(raw_array, (int(w * scale), int(h * scale)), interpolation=interpolation)
    else:
        resized_array = raw_array
    succ, enc_img = cv2.imencode('.jpg', resized_array, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if succ:
        return enc_img.tobytes()