## 1. 问题
- 安装Tensorflow遇到'wrapt'包无法安装的错误，信息如下：
`ERROR: Cannot uninstall 'wrapt'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.`

## 2. 解决
- `pip install -U --ignore-installed wrapt enum34 simplejson netaddr`，全局安装时去掉`-U`参数，用户目录安装时保留，改为`sudo pip install --ignore-installed wrapt enum34 simplejson netaddr`
- `sudo pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple`安装tensorflow即可
