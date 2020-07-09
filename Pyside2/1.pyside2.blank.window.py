#!/usr/bin/python3
# 代码参考Z州的先生博客：https://zmister.com/archives/1018.html

#coding:utf-8

from PySide2 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)  # 实例化一个QApplication应用程序，用于初始化GUI
gui = QtWidgets.QMainWindow()           # 实例化一个主窗口
gui.show()                              # 显示主窗口
sys.exit(app.exec_())                   # 当GUI产生退出信号时Python程序结束
