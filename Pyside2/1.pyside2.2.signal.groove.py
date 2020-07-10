#!/usr/bin/python3

# 连接信号与槽的代码
# element.signal_name.connect(slot_name)
# 某element（对象）.的signal（事件函数）.connect到某个槽（执行函数）

from PySide2 import QtWidgets       # 导入窗体部件

app = QtWidgets.QApplication()      # 初始化一个Q程序的实例，命名为app

btn_quit = QtWidgets.QPushButton("Quit")    # 创建一个按钮（显示Quit），命名为btn_quit

btn_quit.resize(100, 80)            # 设置btn_quit对象的size属性，用resize重置其窗口大小

btn_quit.clicked.connect(app.quit)  # 将btn_quit对象的click事件，连接到app实例的quit执行函数

btn_quit.show()                     # 定义完成后，显示该按钮

app.exec_()                         # 运行该程序
