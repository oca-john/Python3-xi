#!/usr/bin/python3

# input function
name = input("please input your name")  # 提示用户输入内容，并赋值给name变量
print("your name is:",name)             # 通过print函数打印结果

# format output - %占位符格式化
name = 'zhang'                          # 定义内容变量
age = 29
print("your name is %s, age is %i!" %(name, age))       # 文本区域内用%s/%i代替可变区，变量区域内用%()放置所有变量
                                        # %s指string字串，%i指int整型变量

# format output - format格式化
name = 'zhang'                          # 定义内容变量
age = 29
print("your name is {}, age is {}!".format(name, age))  # 文本区域内用{}代替可变区，变量区域内用.format()放置所有变量

# function define
