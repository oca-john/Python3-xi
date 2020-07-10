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


'''
有很多控件都直接或间接的继承了QWidget，所以重点了解下这个类里面都有什么重要的函数。
首先QWidget里面包含了很多事件函数，比如像什么窗口调整，或者窗口显示消失，又或者鼠标进入窗口，离开窗口等等，都会调用一系列的事件函数。

一些事件函数:
resizeEvent(event)      窗口大小调整事件
showEvent(event)        窗口显示事件
hideEvent(event)        窗口隐藏事件
enterEvent(event)       鼠标进入窗口事件
leaveEvent(event)       鼠标离开窗口事件
mouseMoveEvent(event)   鼠标在窗口移动事件，注意如果要使这个生效，需要调用setMouseTracking(true)

除了一些事件，还有一些槽函数：
close()                 关闭窗口
hide()                  隐藏窗口
show()                  显示窗口
update()                刷新窗口

还有一些重要的函数，比如：
resize(w,h)             设置窗口大小
setFixedSize(w,h)       设置窗口固定大小
setMinimumSize(minw, minh)  设置窗口最小大小
setLayout(arg)          设置窗口里面的布局
setWindowFlags(type)    设置窗口选项，比如窗口风格
size()                  返回窗口大小

还有很多，具体请查阅文档。
所以当一个控件继承了QWidget的时候，上面所有的东西都会自动继承，我们可以利用已有的控件，组合成新的控件来进行复用。
'''
