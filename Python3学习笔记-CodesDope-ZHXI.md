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

```markdown
0. 工作方式，写脚本，运行
1. 简单的打印方式,语法元素（数字、字符串、变量、运算符、转义符）
2. 语法结构（顺序执行、循环结构（for、while）、选择结构（if）、跳过与终止）
3. 文件读写
4. 正则表达式
5. 定义函数、调用函数
6. 组织模块、调用模块
7. 类、对象、继承性
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

### 1.4 String

```python
a = 'hello'
print(a[0])							# get 'h'
print(len(a))						# get 5
print(a[1:4])						# get 'ell', not include 'o'[4]
____________________________________
import string
b = 'hello world'
print(a.split())					# get ('hello','world'), split by space
print(a.split('o'))					# get ('hell',' w','rld'), split by 'o'
print('-'.join(b))					# join the words, by '-'
____________________________________
print(a.capitalize(b))				# capitalize the string
print(a.lower(b))					# lowercase the string
print(a.upper(b))					# uppercase the string
print(a.swapcase(b))				# lower -> upper, upper -> lower
```

### 1.5 List, Tuples

```python
a = [1,2,3,4]						# define list by []
print(type(a))						# type() get the class of 'list'
print(a[0])							# pick the ele by index[0]
a[1] = 5							# [1,5,3,4], change the ele
b = [1,2,"anything",4.0]			# list can include diff kinds of ele
____________________________________
print(list(range(0)))				# [], 0 ele from 0 -> none ele
print(list(range(10)))				# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10 ele from 0
print(list(range(1,11)))			# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(range(1,10,2)))			# [1, 3, 5, 7, 9], 2 is step
____________________________________
c = list(range(1,5))				# define a list by list() and range()
print(len(c))						# get the length by len()
____________________________________
d = c[0:3]							# [1,2,3]
____________________________________

```

```python
a = (1,2,3)							# define tuple by (), can't change the ele
```



### 1.6 Dict



## 2. Structure

### 2.1 if...else...

```python
a = int(input())
if a > 10:
	print ("your num is greater than 10\n")
elif a == 10:
    print ("your num is eq to 10\n")
else:
    print ("your num is less than 10\n")
```

### 2.2 while



### 2.3 For



## 3. File and Directory I/O

```python
open(file, 'r')
```



## 4. Regular expressions

```python

```



## 5. Function

```python

```



## 6. Modules

```python

```



## 7. Classes, Objects, Inheritance

```python

```

