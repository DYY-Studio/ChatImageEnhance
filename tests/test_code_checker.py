import ast
import unittest

from sandbox.code_checker import (
    AgentCodeChecker,
    SecurityViolation,
    find_model_loader_calls_missing_local_files_only,
    validate_model_loaders_local_files_only,
)


class AgentCodeCheckerClassSupportTests(unittest.TestCase):
    def test_allows_helper_module_class_with_super_init(self):
        code = """
import torch

class SRVGGNetCompact(torch.nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)

def safe_sr(img, cache=None, device='cpu'):
    return img
"""
        AgentCodeChecker().visit(ast.parse(code))

    def test_allows_two_argument_super_init(self):
        code = """
import torch

class SRVGGNetCompact(torch.nn.Module):
    def __init__(self):
        super(SRVGGNetCompact, self).__init__()

def safe_sr(img, cache=None, device='cpu'):
    return img
"""
        AgentCodeChecker().visit(ast.parse(code))

    def test_still_blocks_other_dunder_attributes(self):
        code = """
def safe_escape(img):
    return (1).__class__
"""
        with self.assertRaises(SecurityViolation):
            AgentCodeChecker().visit(ast.parse(code))

    def test_blocks_non_init_dunder_method_definitions(self):
        code = """
class Unsafe:
    def __getattribute__(self, name):
        return name

def safe_escape(img):
    return img
"""
        with self.assertRaises(SecurityViolation):
            AgentCodeChecker().visit(ast.parse(code))


class ModelLoaderOfflineTests(unittest.TestCase):
    def test_flags_from_pretrained_without_local_files_only(self):
        code = """
from transformers import AutoModel

def safe_model(img, cache=None, device='cpu'):
    model = AutoModel.from_pretrained('org/model')
    return img
"""
        missing = find_model_loader_calls_missing_local_files_only(code)

        self.assertEqual(missing, [(5, "AutoModel.from_pretrained")])
        with self.assertRaisesRegex(ValueError, "local_files_only=True"):
            validate_model_loaders_local_files_only(code)

    def test_accepts_direct_local_files_only_keyword(self):
        code = """
from transformers import AutoModel

def safe_model(img, cache=None, device='cpu'):
    model = AutoModel.from_pretrained('org/model', local_files_only=True)
    return img
"""
        self.assertEqual(find_model_loader_calls_missing_local_files_only(code), [])
        validate_model_loaders_local_files_only(code)

    def test_accepts_nested_pipeline_model_kwargs(self):
        code = """
from transformers import pipeline

def safe_model(img, cache=None, device='cpu'):
    pipe = pipeline('image-to-image', model='org/model', model_kwargs={'local_files_only': True})
    return img
"""
        self.assertEqual(find_model_loader_calls_missing_local_files_only(code), [])
        validate_model_loaders_local_files_only(code)

    def test_flags_modelscope_pipeline_without_local_files_only(self):
        code = """
from modelscope.pipelines import pipeline

def safe_model(img, cache=None, device='cpu'):
    pipe = pipeline('image-denoising', model='iic/model')
    return img
"""
        missing = find_model_loader_calls_missing_local_files_only(code)

        self.assertEqual(missing, [(5, "pipeline")])

    def test_flags_imported_pipeline_alias_without_local_files_only(self):
        code = """
from transformers import pipeline as hf_pipeline

def safe_model(img, cache=None, device='cpu'):
    pipe = hf_pipeline('image-to-image', model='org/model')
    return img
"""
        missing = find_model_loader_calls_missing_local_files_only(code)

        self.assertEqual(missing, [(5, "hf_pipeline")])

    def test_ignores_local_pipeline_helper(self):
        code = """
import torch

def pipeline(value):
    return value

def safe_model(img, cache=None, device='cpu'):
    return pipeline(img)
"""
        self.assertEqual(find_model_loader_calls_missing_local_files_only(code), [])


if __name__ == "__main__":
    unittest.main()
