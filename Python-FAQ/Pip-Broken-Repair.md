
## 1. 问题引入
- 因为使用conda，导致原生Python环境中的pip损坏，调用或升级时出现`cannot import name "FormatControl"`问题；
- 不排除其他原因导致该问题出现；

## 2. 问题解决
### 2.1 退出conda环境
- 注释`.bashrc`文件中conda相关的行（kate中Ctrl+d即可）
- `source .bashrc`刷新shell配置
- 重新进入terminal会话

### 2.2 确认损坏的pip
- `python -m ensurepip --default-pip`

### 2.3 下载pip安装脚本
- `wget https://bootstrap.pypa.io/get-pip.py`

### 2.4 安装pip
- `python3 get-pip.py`
- `pip --version`测试
