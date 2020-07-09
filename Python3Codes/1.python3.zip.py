#!/usr/bin/python3

# 定义3个数组
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]

# 以迭代/交错方式从a,b数组中读取内容，组成新的对象
zipped = zip(a,b)           # 返回一个对象
zipped                      # 这个对象存在内存中，无法直接查看
#>> <zip object at 0x103abc288>
list(zipped)                # list()转换为列表可以查看
[(1, 4), (2, 5), (3, 6)]
list(zip(a,c))              # a,c数组长短不一，元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]

# zip的逆向还原zip(*)
a1, a2 = zip(*zip(a,b))     # 与zip相反，zip(*)可理解为解压，返回二维矩阵式
list(a1)
[1, 2, 3]
list(a2)
[4, 5, 6]
