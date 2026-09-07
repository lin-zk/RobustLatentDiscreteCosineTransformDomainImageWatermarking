# -*- coding: utf-8 -*-
# @Author    : Lin_zk
# @E-mail    : 1751740699@qq.com; eezhengkanglin@mail.scut.edu.cn
# @Thanks to : Copilot for assistance

import os
import importlib

# 动态导入当前目录下的所有模块
__all__ = []
current_dir = os.path.dirname(__file__)
for file in os.listdir(current_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        module = importlib.import_module(f".{module_name}", package=__name__)
        __all__.append(module_name)