#!/usr/bin/python3
# iter()是迭代器，用于迭代遍历对象中的每个元素

lst = [2,3,4]
for i in iter(lst):   # for循环的in后面应该接一个列表或范围，iter()从前面定义的list中生成元素，逐个带入for循环
  print(i)
