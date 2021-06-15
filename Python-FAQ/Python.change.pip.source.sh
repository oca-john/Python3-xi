#!/usr/bin/sh

# 临时使用国内pip源
pip install pkgs -i https://mirrors.ustc.edu.cn/pypi/web/simple

# 永久修改为国内pip源
pip install pip -U pip    (实际上等同于：pip install pip --upgrade pip)
pip config set global.source https://mirrors.ustc.edu.cn/pypi/web/simple
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
