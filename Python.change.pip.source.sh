#!/usr/bin/sh

# 临时使用国内pip源
pip install pkgs -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久修改为国内pip源
pip install pip -U pip    (实际上等同于：pip install pip --upgrade pip)
pip config set global.source https://pypi.tuna.tsinghua.edu.cn/simple
