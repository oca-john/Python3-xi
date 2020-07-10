#!/usr/bin/python3

from PySide2 import QtWidgets, QtCore       # 导入窗体部件和核心库

app = QtWidgets.QApplication()              # 初始化一个Q程序的实例，app

window = QtWidgets.QWidget()                # 创建一个窗口对象，window

slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, window)    # 创建Qslider对象，显示为横向，父类是window
slider.resize(200, 140)                     # 滑动条占用的空间resize一下
slider.setMaximum(100)                      # 设置滑动条显示长度的最大值
slider.setMinimum(0)                        # 最小值

label = QtWidgets.QLabel('None', window)    # 创建Qlabel对象，默认显示"None"，父类是window

slider.valueChanged.connect(label.setNum)   # 滑动条对象的值改变，连接到label对象的显示值改变

window.show()                               # 显示window

app.exec_()                                 # 运行该程序
