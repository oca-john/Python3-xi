#!/usr/bin/python3
# f'{}'格式化输出测试

fst = 'john'
lst = 'smith'
msg = f'{fst} [{lst}] is a coder.'    # 整体语法是用f''产生格式化字符串，其中每个{}都是一个空位，中间填入前面的变量，会自动代换
print(msg)
