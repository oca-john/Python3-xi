#!/usr/bin/python3
# 涉及对象的定义过程，不能交互式执行，需要放入.py代码文件中执行。

# 导入LCD数字，滑块，部件，Box布局，Q程序，网格布局
from PySide2.QtWidgets import QLCDNumber, QSlider, QWidget, QVBoxLayout, QApplication, QGridLayout
# 导入Qt库
from PySide2.QtCore import Qt

class MyLCDNumber(QWidget):                 # 创建LCD数字显示器类
    def __init__(self, parent=None):        # 初始化，无父类
        super().__init__(parent)

        self.lcd_number = QLCDNumber()      # 创建一个lcd数字显示器对象
        self.slider = QSlider(Qt.Horizontal)# 创建滑动条，水平显示

        self.layout = QVBoxLayout()         # 两元素使用垂直布局（上下排列）
        self.layout.addWidget(self.lcd_number)  # 将lcd_num对象加入
        self.layout.addWidget(self.slider)  # 将slider对象加入

        self.setLayout(self.layout)
        self.setFixedSize(120, 100)         # 设置整个控件大小

        self.lcd_number.setDigitCount(2)    # 设置lcd显示器最多显示两位数字
        self.slider.setRange(0, 99)         # 设置可调节的范围
        self.slider.valueChanged.connect(self.lcd_number.display)   # 滑动条的值修改，连接到lcd的显示值

app = QApplication()                        # 初始化Q程序实例，app

window = QWidget()                          # 创建window实例，继承自Q部件

layout = QGridLayout()                      # 布局使用网格布局

mylcdnumber01 = MyLCDNumber()               # 创建lcd显示器的4个实例
mylcdnumber02 = MyLCDNumber()
mylcdnumber03 = MyLCDNumber()
mylcdnumber04 = MyLCDNumber()

layout.addWidget(mylcdnumber01, 1, 1)       # 将4个lcd显示器实例，逐个加入到全局控件中（按照坐标）
layout.addWidget(mylcdnumber02, 1, 2)
layout.addWidget(mylcdnumber03, 2, 1)
layout.addWidget(mylcdnumber04, 2, 2)

window.setLayout(layout)                    # window对象使用上述layout布局

window.show()                               # 显示window对象

app.exec_()                                 # 执行程序
