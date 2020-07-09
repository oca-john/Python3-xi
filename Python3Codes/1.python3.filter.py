#!/usr/bin/python3
# filter是个过滤函数

# 定义函数用于过滤
def is_odd(n):            # 判断是否为奇数
    return n % 2 == 1
tmplist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # 类似map()用法，用函数遍历筛选列表
newlist = list(tmplist)   # filter结果和map结果一样，都是内存地址，需用list()转换为列表才可展示
print(newlist)


import math
def is_sqr(x):            # 用math包中的函数定义可开(整)平方的数
    return math.sqrt(x) % 1 == 0
tmplist = filter(is_sqr, range(1, 101))                     # 用函数遍历筛选范围
newlist = list(tmplist)                                     # list()转换为列表
print(newlist)
