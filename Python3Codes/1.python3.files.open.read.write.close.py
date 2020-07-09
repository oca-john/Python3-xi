#!/usr/bin/python3
# python3.files.open.read.write.close

# open file
f = open("/home/user/documents/filename.txt", "r")  # open文件给句柄f
# read content of file
str = f.read()                                      # 用句柄调用文件，用read()函数读取内容给变量str
print(str)                                          # 打印变量str内容
# close file
f.close()                                           # f.close()关闭句柄

# open file
f = open("/home/user/documents/filename.txt", "r")  # open文件给句柄f
# write content into file
f.write("python is useful, I'll learn it.\n")       # f.write()写入内容
# close file
f.close()                                           # 关闭句柄
