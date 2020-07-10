#!/usr/bin/python3

import PySide2.QtCore           # 此案例只用的"qt.core部分"
from PySide2.QtWidgets import QApplication, QLabel  # 具体用到的是qtwidgets部件中的"Q程序和Q标签"

print(PySide2.__version__)      # 查看PySide2的版本信息，和torh.__version__类似

app = QApplication()            # 初始化Q程序类的一个实例，命名为app

label = QLabel('Hello world!')  # 为程序添加部件Qlabel标签对象，命名为label

label.show()                    # 显示label标签对象

app.exec_()                     # 程序退出时结束
