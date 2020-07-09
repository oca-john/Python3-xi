#!/usr/bin/python3
# dict() is used to create a dict object

# 创建空字典
dict()

# 赋值方式构造字典
dict(one=1, two=2, three=3)               # 赋值符前面的是变量所以可不加引号，后面是值（数值不加引号）
>> {'one': 1, 'two': 2, 'three': 3}

# 映射方式/锯齿方式构造字典
dict(zip(['one','two','three'],[1,2,3]))  # zip()包含两个数组，一对一映射方式组成键值对
>> {'one': 1, 'two': 2, 'three': 3}

# 可迭代对象方式构造字典
dict([('one',1), ('two',2), ('three',3)]) # []包含字典内容，每个键值对单独用()包裹
>> {'one': 1, 'two': 2, 'three': 3}
