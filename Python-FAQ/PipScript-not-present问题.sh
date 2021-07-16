#!/usr/bin/sh
# 若出现：Script file 'D:\Programs\miniconda3\envs\mindspore\Scripts\pip-script.py' is not present.报错：
# 重新下载pip安装脚本并安装即可
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
