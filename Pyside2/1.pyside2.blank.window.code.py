#!/usr/bin/python3
# 代码参考Z州的先生博客：https://zmister.com/archives/1018.html
# 此版本是基于代码流的，和用户操作流程符合，小型软件方便写，但是大型软件不易维护。
# coding:utf-8

from PySide2 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)  # 实例化一个"QApplication程序"，初始化GUI
gui = QtWidgets.QMainWindow()           # 实例化一个"QMainWindow主窗口"
gui.show()                              # "显示主窗口"
sys.exit(app.exec_())                   # 当"GUI产生退出信号"时Python程序结束
