#!/usr/bin/python3
# 代码参考Z州的先生博客：https://zmister.com/archives/1018.html
# 此版本是面向对象版本，若程序功能增多、控件增加，也更容易进行优化和维护。
# coding:utf-8

from PySide2 import QtWidgets
import sys

class App(QtWidgets.QMainWindow):           # 创建App类，初始化主窗口
    def __init__(self):
        super().__init__()

def main():
    app = QtWidgets.QApplication(sys.argv)  # 创建Q程序app，初始化
    gui = App()                             # gui是App类的实例化
    gui.show()                              # show()方法显示窗口
    sys.exit(app.exec_())                   # 调用系统中的退出命令

if __name__ == '__main__':
    main()
