#!/usr/bin/python3
# super()用于调用父类的方法

class A:                # 定义A类
     def add(self, x):  # 定义父类A的add方法（初始化self参数（必须），x参数）
         y = x+1        # 定义公式
         print(y)
class B(A):             # 定义B类，以A为父类
    def add(self, x):   # 定义子类B的add方法（初始化self参数（必须），x参数）
        super().add(x)  # super()调用父类A的方法，来定义子类的方法，类似于继承
b = B()                 # 创建子类B的一个实例b
b.add(2)                # 实例b，调用子类B的方法add，间接调用了父类A的方法add，本质就是y=x+1
# 传入的参数是2,实例b的输出就是3
