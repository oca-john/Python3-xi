# Cython 编译器的初始化和基本使用

## Cython 编译器介绍
Cython 是一门单独的语言（以 .pyx 为扩展名），也是 Python 的一个扩展，用于使 Python 具有调用 C-API 的能力，将 Python 代码编译为 C 代码以及可执行文件。

## Cython 编译器安装
使用 pip install cython 安装即可，会根据 pip 调用的 Python 版本确定安装的相应版本。

## 编译过程注意
Cython 默认调用 Python2 接口（即参数 -2），如使用 Python3，需要以 'cython -3 name.pyx' 进行编译。

## 方法一 使用静态对象编译为so可执行文件
### 1. 创建 Cython 脚本
``` cython
print "Hello World"             # 交互式输出
def hello():                    # 函数式输出
    print 'hello in hello.pyx'
```

### 2. 创建 setup.py 脚本
``` cython
from distutils.core import setup    # 导入核心工具包中的安装工具
from Cython.Build import cythonize  # 导入编译工具
setup(                              # 设置编译参数
    ext_modules = cythonize("hello.pyx")
)
```

### 3. 根据编译参数编译脚本
``` cython
python setup.py build_ext --inplace
```

### 4. 调用编译后的 .so 静态库
``` cython
import hello
hello.hello()
```

## 方法二 使用三方库
### 5. 调用 pyximport 包
``` cython
import pyximport                # 导入关键包
pyximport.install()             # 安装

import hello                    # 导入上述自定义的 hello 包
hello.hello()                   # 输出
```
