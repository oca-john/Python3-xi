#!/usr/bin/python3

# 用python实现python的开发环境的初始化配置
# 主要包括默认python和pip版本配置，相关三方库的安装，深度学习相关库的安装

# 导入相关库
import os

# 1. 版本配置
os.system("cd /usr/bin/; sudo rm python; sudo ln -s python3.* /usr/bin/python; sudo rm pip; sudo ln -s pip3 /usr/bin/pip; cd;")
print("python3, pip3 were configured as python & pip.")

# 2. 三方库
os.system("sudo zypper in python3-numpy python3-scipy python3-matplotlib python3-seaborn;")
print("3rd part lib were configured well.")

# 3. 深度学习库
os.system("sudo pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -i https://pypi.tuna.tsinghua.edu.cn/simple")
print("deeplearning lib pytorch was installed well.")
