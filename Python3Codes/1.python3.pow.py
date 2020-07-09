#!/usr/bin/python3
# pow()有内置的，也有math包中的
# pow()主要计算x^y(x的y次方)

# 内置pow()函数
pow(x,y[,z])      # 其中的z参数是可选的，用于对x的y次方的结果取模，等于`pow(x,y)%z`

pow(2,3)
#>> 8             # 2的3次方为8
pow(2,3,3)
#>> 2             # 2的3次方为8,8%3为2(6%3=0)

# math包中的pow()函数
import math
math.pow(x,y)

math.pow(2,3)
#>> 8.0           # 结果为8,math模块会将结果转换为float类型
