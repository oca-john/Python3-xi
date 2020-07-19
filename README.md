# Python3-xi

Python3是目前应用领域最多最流行的脚本语言，有多方面潜力，在深度学习方面暂时居于不可替代的地位。

Pytorch非常流行的一个原因，没有完全将张量的操作API化，让用户有足够的操作深度和自由度，可以自己掌控开发进度，也可以方便的融入别人的项目。

Python3 related libs:  
> syntax-check: **pylint**;  
> intelligent-syntax-env: **pylance**;  
> basic-libs: **numpy**, **pandas**, **matplotlib**;  
> advanced: **seaborn**, **scipy**;  
> ai-related: **torch**, **torchvision**;  
> GUI-dev: **pyside2**;  

## VScode中新建Jupyter-notebook文件
1. 安装VScode编辑器，官方方案太慢，推荐下载最新安装包自己手动安装（windows下推荐软件内升级）。  
2. 安装Python扩展。  
3. 创建新的Notebook笔记本，`Ctrl + Shift + p`打开命令栏，输入`jupyter`，选择`creat new blank jupyter notebook`创建笔记本。


## 关于jupyter_notebook的初始目录

在windows平台，舒服地在本地使用jupyter_notebook还是比较麻烦的，记录一下使用技巧，方便自己日后使用。

> 本段主要是为了在本地以指定目录为工作目录，打开jupyter_notebook浏览器（因为该浏览器不支持向上退回至根目录），以管理或阅读目录下的所有ipynb文件。

### a. 定位到指定目录打开Jupyter
1. 打开本地Console，cd定位到指定目录（保存jupyter笔记本的目录）
2. 手动`jupyter notebook`打开notebook浏览器【可以中间用空格】

### b. 将用户Documents文件夹设为默认目录
1. 打开conda所在文件夹，调出conda快捷图标的属性（非cmd属性）
2. 修改`Start in`起始位置属性为`D:\Users\User\Documents`，即为Jupyter默认位置
3. 打开conda，输入`jupyter-notebook`命令打开notebook浏览器【也可以中间用连字符】

## vscode.code-runner.wsl-linux
- WSl2中用vscode调用默认终端实现高效开发

### 软件环境
控制面板中开启wsl功能，微软商店安装linux发行版。  
- 安装完创建普通用户并设置密码；  
- 安装基本的开发工具（linux端）；  

安装vscode软件以及remote-wsl, code-runner插件。  

### 工具连接
#### 发行版和默认终端解释器
打开vscode，在remote-wsl插件中选择默认的发行版，在terminal中选择默认解释器为wsl（即指向特定发行版的bash）。  

#### 编辑器运行和终端解释器
编辑器部分默认采用cmd，搜寻windows下的语言环境。  
F1打开设置搜索框-搜索setting.json （文件-预设-设置-搜索setting.json-edit.in.setting.json）。  

#### setting-json编辑
``` json
    "code-runner.defaultLanguage": "python"		# 插件使用的默认语言
    "code-runner.runInTerminal": true,			# 编辑器代码是否用终端解释
    "code-runner.terminalRoot": "/mnt/",		# 终端解释器的root目录
```
## pyside2
### Python最“自由”的三方GUI库
Pyhton作为最流行的编程语言之一，具有相当多的三方GUI库来辅助其图形开发，除了自身所带的Tk库之外，Wx库、Gtk库、PyQt库是最著名的。  

Tk库灵活性和功能性相对于三方库相去甚远，Gtk是Gnome背后的底层库，基于C语言，其Python绑定据说较复杂，主要是C语言三方库的缺少导致一些功能的实现较困难。  

Wx库和Qt库都是基于C++的(后者是KDE的底层)，区别在于前者是纯社区驱动的，API、开发文档不甚完善；后者经历过商业、半商业（LGPL）的演变，API和文档非常完善，开发实例也多。两者对商业软件的开发都比较友好，允许开发私有应用而不必公开源码，只需要标注引用了那些LGPL标准库，若没有采用官方的标准库，则连引用都不需要。  

在Qt被诺基亚收购后，诺基亚尝试让PyQt更自由，希望其所有公司使用LGPL发布其源码，但被拒绝，于是有了开源世界与之对等的Qt for Python项目，也就是后来的PySide和PySide2库。PySide2库采用了LGPL协议，天生对商业用户友好，加之与PyQt的兼容性超高，因此是Python GUI开发的最佳方案。  

> 环境配置
>> 优先考虑openSUSE上开发，可避免windows上Qt只能用商业版本的问题，若纯命令开发则不存在该问题。  
>> software.opensuse.org安装Qt  
>> `sudo zypper in python3-pyside2`  
>> `sudo pip install pyside2`  

## Qt & Qt-creator
Qt是KDE的基石，目前采用LGPL协议，是GUI绘制的不错选择。Gtk、Wx等虽都开源，但开发速度和易用性较Qt仍然略差。  
Qt在Linux下直接从YaST中安装`patterns-libqt5`，在Windows下直接从清华源下载，离线安装不会出现订购页面。  
Qt的Python开发流程。在Qt-creator中新建项目，选择`Qt for Python - Window`开发桌面应用。  

项目中`.ui`文件用于绘制GUI负责操作逻辑，而`.py`代码文件用于写业务逻辑。  
写完后的`.ui`文件本质上是`.xml`文件，需要用官方的`pyside2-uic`工具将标记语言转化为Python绘制GUI的代码。  
之后在业务逻辑`.py`文件中`import`引用该文件即可。
最终程序直接用`.py`文件执行。
