#!/usr/bin/python3
# range()返回可迭代对象，不是列表类型(在python2中是列表类型)
# list()函数是对象迭代器

# range(stop)                 # 约束终止数值(默认从0开始)
# range(start, stop[, step])  # 起止数值和步长

range(5)                      # range()对象是range类型，不是list类型
#>> range(0,5)
list(range(5))                # list()对象是列表
#>> [0, 1, 2, 3, 4]
list(range(0,30,5))           # 正向步长
#>> [0, 5, 10, 15, 20, 25]
list(range(0, -10, -1))       # 负向步长
#>> [0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
