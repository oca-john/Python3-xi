#!/usr/bin/python3

from PySide2 import QtWidgets       # 导入窗体部件

app = QtWidgets.QApplication()      # 初始化一个Q程序实例，app

window = QtWidgets.QWidget()        # 创建窗口对象，window
window.resize(200, 120)             # 重置大小resize

btn_quit = QtWidgets.QPushButton("Quit", window)    # 创建按钮对象，btn_quit

btn_quit.setGeometry(10, 40, 180, 40)   # 设置按钮的大小

btn_quit.clicked.connect(app.quit)  # 按钮对象的clicked信号，连接到app实例的quit函数

window.show()                       # 显示窗体

app.exec_()                         # 执行程序
