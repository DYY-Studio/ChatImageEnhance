import numpy as np
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import get_executable_dir

_REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "tool_registry_under_test",
    PROJECT_ROOT / "tools" / "registry.py"
)
_REGISTRY_MODULE = importlib.util.module_from_spec(_REGISTRY_SPEC)
assert _REGISTRY_SPEC.loader is not None
_REGISTRY_SPEC.loader.exec_module(_REGISTRY_MODULE)
ToolRegistry = _REGISTRY_MODULE.ToolRegistry


def test_runtime_model_dir_injection_uses_source_cache_root():
    seen = {}

    def sample_tool(img: np.ndarray, model_dir: str = "") -> np.ndarray:
        seen["model_dir"] = model_dir
        return img

    schema = {
        "name": "Sample_HF_Tool",
        "description": "sample",
        "parameters": {
            "model_dir": {"type": "str", "description": "runtime injected"}
        },
        "source": "huggingface",
        "repo_id": "org/model-name",
    }
    registry = ToolRegistry()
    registry.dynamic_register(sample_tool, schema)

    img = np.zeros((2, 2), dtype=np.uint8)
    registry.tools["Sample_HF_Tool"]["func"](img)

    expected = (
        get_executable_dir()
        / "caches"
        / "model_assets"
        / "huggingface"
        / "org__model-name"
    )
    assert seen["model_dir"] == str(expected.resolve())


def test_runtime_model_dir_injection_preserves_explicit_path():
    seen = {}

    def sample_tool(img: np.ndarray, model_dir: str = "") -> np.ndarray:
        seen["model_dir"] = model_dir
        return img

    schema = {
        "name": "Sample_ModelScope_Tool",
        "description": "sample",
        "parameters": {
            "model_dir": {"type": "str", "description": "runtime injected"}
        },
        "source": "modelscope",
        "repo_id": "iic/some-model",
    }
    registry = ToolRegistry()
    registry.dynamic_register(sample_tool, schema)

    img = np.zeros((2, 2), dtype=np.uint8)
    registry.tools["Sample_ModelScope_Tool"]["func"](img, model_dir="D:/models/custom")

    assert seen["model_dir"] == "D:/models/custom"


def test_repo_id_loader_without_model_dir_is_not_forced():
    seen = {}

    def sample_tool(img: np.ndarray, device: str = "cpu") -> np.ndarray:
        seen["device"] = device
        return img

    schema = {
        "name": "Sample_Repo_Id_Tool",
        "description": "sample",
        "parameters": {
            "device": {"type": "str", "description": "runtime injected"}
        },
        "source": "huggingface",
        "repo_id": "org/model-name",
    }
    registry = ToolRegistry()
    registry.dynamic_register(sample_tool, schema)

    img = np.zeros((2, 2), dtype=np.uint8)
    registry.tools["Sample_Repo_Id_Tool"]["func"](img, device="cpu")

    assert seen["device"] == "cpu"


def test_runtime_model_dir_ignores_persisted_absolute_path_when_source_exists():
    schema = {
        "source": "modelscope",
        "repo_id": "iic/some-model",
        "model_dir": "E:/cached/download",
    }

    expected = (
        get_executable_dir()
        / "caches"
        / "model_assets"
        / "modelscope"
        / "iic__some-model"
    )
    assert ToolRegistry.resolve_model_dir(schema) == str(expected.resolve())


if __name__ == "__main__":
    test_runtime_model_dir_injection_uses_source_cache_root()
    test_runtime_model_dir_injection_preserves_explicit_path()
    test_repo_id_loader_without_model_dir_is_not_forced()
    test_runtime_model_dir_ignores_persisted_absolute_path_when_source_exists()
