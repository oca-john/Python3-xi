#!/usr/bin/python3

# slice(stop)                 # 约束终止数值(默认从0开始)
# slice(start, stop[, step])  # 起止数值和步长

myslice = slice(5)            # '截取前5个元素为新的子字串'的切片（类似函数）
myslice
#>> slice(None, 5, None)      # 查看该切片的属性

arr = range(10)               # 创建新数组
arr
#>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr[myslice]                  # 对arr数组使用myslice切片
#>> [0, 1, 2, 3, 4]
