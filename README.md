# Python3-xi

Python3 是目前应用领域最多最流行的脚本语言，有多方面潜力，在深度学习方面暂时居于不可替代的地位。

Pytorch 非常流行的一个原因，没有完全将张量的操作 API 化，让用户有足够的操作深度和自由度，可以自己掌控开发进度，也可以方便的融入别人的项目。

## 1. Python3 related libs:  
> intelligent-syntax-env: **pylance**;  
> basic-libs: **numpy**, **pandas**, **matplotlib**, **seaborn**, **scipy**, **sklearn**;  
> nn-related: **mne**, **neo**;  
> ai-related: **torch**, **torchvision**, **tensorflow(with tensorboard)**;  
> GUI-dev: **pyside2**;  

## 2. VScode 中新建 Jupyter-notebook 文件
1. 安装 VScode 编辑器，官方方案太慢，推荐下载最新安装包自己手动安装（windows 下推荐软件内升级）。  
2. 安装 Python 扩展。  
3. 创建新的 Notebook 笔记本，`Ctrl + Shift + p`打开命令栏，输入`jupyter`，选择`creat new blank jupyter notebook`创建笔记本。


## 3. 关于 jupyter_notebook 的初始目录(优先使用 jupyter-lab)

在 windows 平台，舒服地在本地使用 jupyter_notebook 还是比较麻烦的，记录一下使用技巧，方便自己日后使用。

> 本段主要是为了在本地以指定目录为工作目录，打开 jupyter_notebook 浏览器（因为该浏览器不支持向上退回至根目录），以管理或阅读目录下的所有 ipynb 文件。

### a. 定位到指定目录打开 Jupyter
1. 打开本地 Console，cd 定位到指定目录（保存 jupyter 笔记本的目录）
2. 手动`jupyter notebook`打开 notebook 浏览器【可以中间用空格】

### b. 将用户 Documents 文件夹设为默认目录
1. 打开 conda 所在文件夹，调出 conda 快捷图标的属性（非cmd属性）
2. 修改`Start in`起始位置属性为`D:\Users\User\Documents`，即为 Jupyte r默认位置
3. 打开 conda，输入`jupyter-notebook`命令打开 notebook 浏览器【也可以中间用连字符】

## 4. vscode.code-runner.wsl-linux
- WSl2中用vscode调用默认终端实现高效开发

### 软件环境
控制面板中开启 wsl 功能，微软商店安装 linux 发行版。  
- 安装完创建普通用户并设置密码；  
- 安装基本的开发工具（linux端）；  

安装 vscode软件以及 remote-wsl, code-runner 插件。  

### 工具连接
#### 发行版和默认终端解释器
打开 vscode，在 remote-wsl 插件中选择默认的发行版，在 terminal 中选择默认解释器为 wsl（即指向特定发行版的 bash）。  

#### 编辑器运行和终端解释器
编辑器部分默认采用 cmd，搜寻 windows 下的语言环境。  
F1 打开设置搜索框-搜索 setting.json （文件-预设-设置-搜索 setting.json-edit.in.setting.json）。  

#### setting-json 编辑
``` json
    // VScode 设置
    "workbench.iconTheme": "vscode-icons",    
    "update.mode": "manual",
    "workbench.activityBar.visible": true,


    // Code-runner 设置
    "code-runner.defaultLanguage": "python"             // 默认 code-runner 编程语言为 Python
    "code-runner.runInTerminal": true,                  // 调用终端运行 code-runner 命令
    // "code-runner.terminalRoot": "/mnt/",             // 终端的默认起始位置
    

    // 设置 wsl 解释器
    // "terminal.integrated.shell.windows": "C:\\Windows\\System32\\wsl.exe",   // when use wsl as terminal
    // "python.pythonPath": "D:\\Programs\\Miniconda3\\envs\\tf1\\python.exe",  // tf1 环境中的解释器


    // 设置为 Conda env 中的解释器
    "terminal.integrated.shell.windows": "C:\\Windows\\System32\\cmd.exe",
    "python.pythonPath": "D:\\ProgramData\\Anaconda3\\envs\\fortrain\\python.exe",
    "terminal.integrated.shellArgs.windows": ["/K",
    "D:\\Programs\\miniconda3\\Scripts\\activate.bat D:\\Programs\\miniconda3\\envs\\tf1"]

```

## 5. Miniconda 配置
用 pip 或 conda 安装所有三方库即可。  
尽量避免混合使用 pip 和 conda 命令，虽然二者在 conda 内部共享软件列表。  
pip 安装尽量使用本地清华源：`-i https://pypi.tuna.tsinghua.edu.cn/simple`  
tensorflow (版本号与 python 版本相关)使用本体安装，若由 tensorboard 安装时附带，容易导致 tb 版本比 tf 高的情况，出现报错。用 pip list 或 conda list 查看版本号，再用指定版本的方式 install 一遍以检查依赖关系。【此处需要补充检查命令】

## 5.2 Conda 命令行环境配置
开始前用 `conda info -e` 查看当前所有环境列表，默认只有 base 环境。虚拟环境创建 `conda create -n tf1 python=3.7.5` 创建新的指定python版本的虚拟环境。  
若新环境创建失败，考虑是否是在线仓库连接失败导致的？通过修改软件源到国内的清华源提高连接质量，重新创建。  
`conda activate tf1` 进入该环境；在环境内 `conda deactivate` 则退出至 base 环境。  
在开始菜单中找到 miniconda3 的图标，找到快捷方式位置，属性中可设置两个参数。  
> target 中通过修改默认 conda 指向的文件夹位置，指定 `tf1` 环境所在的文件夹，即可在进入 conda 后默认进入该虚拟环境，而非 base。  
> 修改默认打开的地址，到自己常用的文件夹，如 Documents 文件夹或个人代码目录。
