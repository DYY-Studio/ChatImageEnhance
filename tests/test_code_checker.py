import ast
import unittest

from sandbox.code_checker import AgentCodeChecker, SecurityViolation


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


if __name__ == "__main__":
    unittest.main()
