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
    "code-runner.defaultLanguage": "python"		# 插件使用的默认语言
    "code-runner.runInTerminal": true,			# 编辑器代码是否用终端解释
    "code-runner.terminalRoot": "/mnt/",		# 终端解释器的root目录
```

## 5. Miniconda 配置
用 pip 或 conda 安装所有三方库即可。  

尽量避免混合使用 pip 和 conda 命令，虽然二者在 conda 内部共享软件列表。  

pip 安装尽量使用本地清华源：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

tensorflow (版本号与 python 版本相关)使用本体安装，若由 tensorboard 安装时附带，容易导致 tb 版本比 tf 高的情况，出现报错。用 pip list 或 conda list 查看版本号，再用指定版本的方式 install 一遍以检查依赖关系。【此处需要补充检查命令】

## 5.2 Conda 命令行环境配置
开始前用 `conda info -e` 查看当前所有环境列表，默认只有 base 环境。虚拟环境创建 `conda create -n tf1 python=3.6` 创建新的指定python版本的虚拟环境。  
`conda activate tf1` 进入该环境；在环境内 `conda deactivate` 则退出至 base 环境。  
在开始菜单中找到 miniconda3 的图标，找到快捷方式位置，属性中可设置两个参数。  
> target 中通过修改默认 conda 指向的文件夹位置，指定 `tf1` 环境所在的文件夹，即可在进入 conda 后默认进入该虚拟环境，而非 base
> 修改默认打开的地址，到自己常用的文件夹，如 Documents 文件夹或个人代码目录。


## 6. pyside2
### Python 最“自由”的三方GUI库
Pyhton 作为最流行的编程语言之一，具有相当多的三方 GUI 库来辅助其图形开发，除了自身所带的 Tk 库之外，Wx 库、Gtk 库、PyQt 库是最著名的。  

Tk 库灵活性和功能性相对于三方库相去甚远，Gtk 是 Gnome 背后的底层库，基于 C 语言，其 Python 绑定据说较复杂，主要是 C 语言三方库的缺少导致一些功能的实现较困难。  

Wx 库和 Qt 库都是基于 C++ 的(后者是 KDE 的底层)，区别在于前者是纯社区驱动的，API、开发文档不甚完善；后者经历过商业、半商业（LGPL）的演变，API 和文档非常完善，开发实例也多。两者对商业软件的开发都比较友好，允许开发私有应用而不必公开源码，只需要标注引用了那些 LGPL 标准库，若没有采用官方的标准库，则连引用都不需要。  

在 Qt 被诺基亚收购后，诺基亚尝试让 PyQt 更自由，希望其所有公司使用 LGPL 发布其源码，但被拒绝，于是有了开源世界与之对等的 Qt for Python 项目，也就是后来的 PySide 和 PySide2 库。PySide2 库采用了 LGPL 协议，天生对商业用户友好，加之与 PyQt 的兼容性超高，因此是 Python GUI 开发的最佳方案。  

> 环境配置
>> 优先考虑 openSUSE 上开发，可避免 windows 上 Qt 只能用商业版本的问题，若纯命令开发则不存在该问题。  
>> software.opensuse.org 安装 Qt  
>> `sudo zypper in python3-pyside2`  
>> `sudo pip install pyside2`  

## 7. Qt & Qt-creator
Qt是KDE的基石，目前采用LGPL协议，是GUI绘制的不错选择。Gtk、Wx等虽都开源，但开发速度和易用性较Qt仍然略差。  
Qt在Linux下直接从YaST中安装`patterns-libqt5`，在Windows下直接从清华源下载，离线安装不会出现订购页面。  
Qt的Python开发流程。在Qt-creator中新建项目，选择`Qt for Python - Window`开发桌面应用。  

项目中`.ui`文件用于绘制GUI负责操作逻辑，而`.py`代码文件用于写业务逻辑。  
写完后的`.ui`文件本质上是`.xml`文件，需要用官方的`pyside2-uic`工具将标记语言转化为Python绘制GUI的代码。  
之后在业务逻辑`.py`文件中`import`引用该文件即可。
最终程序直接用`.py`文件执行。
