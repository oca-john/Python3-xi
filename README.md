# Python3-xi

Python3是目前应用领域最多最流行的脚本语言，有多方面潜力，在深度学习方面暂时居于不可替代的地位。

Pytorch非常流行的一个原因，没有完全将张量的操作API化，让用户有足够的操作深度和自由度，可以自己掌控开发进度，也可以方便的融入别人的项目。

Python3 related libs:  
            basic-libs: **numpy**, **pandas**, **matplotlib**;  
            advanced: **seaborn**, **scipy**;  
            ai-related: **torch**, **torchvision**;  

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
