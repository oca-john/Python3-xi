# Python 有用三方库整理

## VScode中的语言服务器: 
pylance;

## 常见的基本库:  
**numpy** ,  **pandas** ,  **matplotlib** , seaborn, scipy, sklearn;

## 神经科学相关：
nn-related:  **mne** , neo;

## 补充几个社区推荐的包: 
`sudo pip install mne-bids nibabel pybv -i https://mirrors.ustc.edu.cn/pypi/web/simple`

也可使用`sudo pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple`命令修改软件源。

## 深度学习相关：
ai-related:  **tensorflow** (with tensorboard), torch, torchvision;

模型转换：onxx(pytorch->onnx), onnx_tf(onnx->tf);

## 用户界面开发相关：
pyside2, (或 **纯Qt** 写UI再与Python绑定);

