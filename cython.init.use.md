# Cython 编译器的初始化和基本使用

## Cython 编译器介绍
Cython 是一门单独的语言（以 .pyx 为扩展名），也是 Python 的一个扩展，用于使 Python 具有调用 C-API 的能力，将 Python 代码编译为 C 代码以及可执行文件。

## Cython 编译器安装
使用 pip install cython 安装即可，会根据 pip 调用的 Python 版本确定安装的相应版本。

## 编译过程注意
Cython 默认调用 Python2 接口（即参数 -2），如使用 Python3，需要以 'cython -3 name.pyx' 进行编译。

## 使用
### 1.创建Cython脚本
``` cython
print "Hello World"
def hello():
    print 'hello in hello.pyx'
```

### 2.
``` cython
from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("hello.pyx")
)
```

### 3.
``` cython
python setup.py build_ext --inplace
```

### 3.
``` cython
import hello
hello.hello()
```

### 3.
``` cython
import pyximport; pyximport.install()

import hello
hello.hello()
```
