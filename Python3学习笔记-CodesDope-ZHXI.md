# Python3学习笔记-CodesDope-ZHXI

- From [https://www.codesdope.com](https://www.codesdope.com/) 

## 0. Introduction

```python
#!/usr/bin/python
print("hello world!")		# py3 need the (), for print is a function
```

1. Run python in the live mode.
   1. Type `python` in your terminal.
   2. Input any command, such as `print ("hello world!")`.
   3. Type `exit()` to exit.
2. Run python in Script mode.
   1. Edit the python script file with Vim or VScode.
   2. cd to the directory which include the script file.
   3. Type `python file_l.py`, to run.

```
0.工作方式，写脚本，运行
1.简单的打印方式
2.语法元素（数字、字符串、变量、运算符、转义符）
3.语法结构（顺序执行、循环结构（for、while）、选择结构（if）、跳过与终止）
4.文件读写
5.正则表达式
6.定义函数、调用函数
7.组织模块、调用模块
8.类、对象、继承性
```

## 1. Elements

### 1.1 Print

```python
print ("hello world!")				# () is need, "" & '' is same, vars translate
print (type('hello world!'))		# print vars' type
print ('hello'*3)					# string 'hello' 3 times
print ('ab'+'cd')					# print string one by one
____________________________________
# print 'fail'						# strings after '#' will be ignored
'''
multiline strings ignored			# use three ' or " to ignore all inner strings
'''
____________________________________
a,b = b,a							# swap two vars' value
```

### 1.2 Input

```python
name = input("what is your name?")	# get the input, as name
print ("your name is ", name, "\n") # use the name directly
age = int(input("how old are you?"))# input get the string, need to be translated
num = float(input("what is the float num?"))
____________________________________
import math
print(math.sin(30))					# do some math with math lib
```

### 1.3 Operators

```python
0, "", undef						# all these three are FALSE
others								# TRUE
and, or, not						# logical AND, OR, NOT
+, -, *, /, %, **					# add, subtract, multiply, divides, modulus, exponent
=									# assigns
+=, -=, *=, /=, %=, **=				# add & assigns, subtract & assigns,...
==, !=, >, <, >=, <=				# compare two values
eq, ne, gt, lt, ge, le				# compare two values
```

## 2. Structure

```python
a = int(input())
if a > 10:
	print ("your num is greater than 10\n")
elif a == 10:
    print ("your num is eq to 10\n")
else:
    print ("your num is less than 10\n")
```

