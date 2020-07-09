#!/usr/bin/python3
# map function is used to map each element in an array run in a certain function
# map函数调用指定函数计算数组中的每个元素的值


# method 1, 先定义后调用
# def a calculating function, 定义一个计算函数
def sqr(x):
  return x**2;
# use map function, 用map函数遍历数组，并用上述函数计算
# map(sqr, [1,2,3]);                      # 这是python2版本，python3中会返回结果的内存地址
list(map(sqr, [1,2,3]));                  # 这是python3版本，加list()会将结果作为list返回

# method 2, 直接用lambda匿名函数定义，并调用
list(map(lambda x: x**2, [1,2,3]));       # 其中`lambda x:x**2`与上述def过程相同，不过使用了匿名函数来定义

# lambda匿名函数定义有两个参数的计算
list(map(lambda x, y: x**2 + y, [1,2,3], [2,4,6])); # 同上定义有两个参数的匿名函数，两个数组进行参数赋值
