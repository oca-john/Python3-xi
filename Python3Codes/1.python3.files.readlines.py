#!/usr/bin/python3
# python3.files.readlines

# open file
f = open("/home/user/documents/filename.txt", "r")  # open文件给句柄f
# read one line each time
# 每次读取一行，换行符为'\n'，如果f.readline()返回空字符，说明已读取到最后一行。
str = f.readline()                                  # read each line
print(str)
# close file
f.close()

# open file
f = open("/home/user/documents/filename.txt", "r")  # open文件给句柄f
# read all lines in the file
str = f.readlines()                                 # read all lines
print(str)
# close file
f.close()

# open file
f = open("/home/user/documents/filename.txt", "r")  # open文件给句柄f
# read content by 'for loop'
for line in f:                                      # read every line
  print(line, end='')
# close file
f.close()
