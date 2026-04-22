import ast

class SecurityViolation(Exception):
    """自定义安全违规异常"""
    pass

class AgentCodeChecker(ast.NodeVisitor):
    def __init__(self):
        # 定义黑名单：禁止直接调用的内置函数
        self.forbidden_names = {'eval', 'exec', 'getattr', 'setattr', 'delattr', 'input', 'open'}
        
    def visit_Import(self, node):
        # 拦截：import os
        raise SecurityViolation("禁止使用 import 语句")

    def visit_ImportFrom(self, node):
        # 拦截：from os import system
        raise SecurityViolation("禁止使用 from...import 语句")

    def visit_Attribute(self, node):
        # 拦截：访问 __subclasses__ 等双下划线私有属性（Python 逃逸常用手段）
        if node.attr.startswith('__'):
            raise SecurityViolation(f"禁止访问私有属性: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        # 拦截：直接调用黑名单中的危险函数
        if node.id in self.forbidden_names:
            raise SecurityViolation(f"禁止使用危险函数: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # 对特定函数调用频率或参数的逻辑检查（占位）
        self.generic_visit(node)