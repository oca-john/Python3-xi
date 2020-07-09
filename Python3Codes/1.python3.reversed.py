#!/usr/bin/python3
# 返回的对象有些是二进制的，需要list()化查看

# 字符串
seqString = 'Runoob'
print(list(reversed(seqString)))
 
# 元组
seqTuple = ('R', 'u', 'n', 'o', 'o', 'b')
print(list(reversed(seqTuple)))
 
# range
seqRange = range(5, 9)
print(list(reversed(seqRange)))
 
# 列表
seqList = [1, 2, 4, 3, 5]
print(list(reversed(seqList)))
