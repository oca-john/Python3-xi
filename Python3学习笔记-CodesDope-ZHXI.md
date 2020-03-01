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
1. 简单的打印方式,语法元素（变量、列表、元组、字典、运算符、转义符）
2. 语法结构（顺序执行、循环结构（for、while）、选择结构（if）、跳过与终止）
3. 文件读写
4. 正则表达式
5. 定义函数、调用函数
6. 类，对象
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
print(c[::2])						# [1,3,5], step is 2, from index 0
print(c[::3])						# [1,4], step is 3
print(c[::-1])						# [5,4,3,2,1], step is -1
print(del(c[1]))					# [1,3,4,5], del the index[1]
____________________________________
e = [[1,2,3],[4,5,6],[7,8,9]]		# 2D list
print(e[1][1])						# 5, 1st_index[1], 2nd_index[1]
____________________________________
f = [1,2]
g = [4,5]
h = f+g								# [1,2,4,5], list one by one
h.append(9)							# [1,2,4,5,9], append in the end
i = f*2								# [1,2,1,2], like 'x' in perl
print(2 in h)						# True, is '2' in list 'h'
____________________________________
j = list(map(int,f))				# [1,2], map all the ele in list
print(f.index(2))					# 1, check the index of ele '2'
```

```python
a = (1,2,"Hello")					# define tuple by ()
a[1] = 12		#TypeError			# can't change the ele
```

| chr  | means      | notice                   |
| ---- | ---------- | ------------------------ |
| []   | list       |                          |
| ()   | tuple      | can't change the element |
| {}   | dictionary | no order                 |

### 1.6 Dictionary

```python
a = {'mango':40,'banana':10}		# define dict by {'key':value}
print(a['mango'])					# 40, access the k-v pair by it's key
a['cherry'] = 20					# add new k-v pair
b = {'a':11, 'b':22, 'c':33}		# diff types are supported
____________________________________
print(list(b.keys()))				# get all keys of dict
print(list(b.values()))				# get all values of dict
print('c' in b.keys())				# is ele 'c' included in dict b
```

## 2. Structure

### 2.1 if...else...

```python
a = int(input("Enter a number"))
if a > 10:
	print ("your num is greater than 10\n")
elif a == 10:
    print ("your num is eq to 10\n")
else:
    print ("your num is less than 10\n")
____________________________________
print("Enter a number")
a = int(input())
if a%2 == 0 and a>10: 				# using 'and' in condition
	print("Your number is greater than 10 and even")
elif a%2 == 0 and a<10: 			# using 'and' to check if both are true
	print("Your number is even and smaller than 10")
else:
	print("Your number is odd")
____________________________________
print("Enter your age.")
age = int(input())
if age < 13:
	print("Hey! kid")
elif age>13 and age < 20:
	pass							# return nothing
else:
	print("You are grown up.")
```

### 2.2 while

```python
i = 1								# timer
while i<=3:
	print(i*4)						# get (4, 8, 12)
	i=i+1							# => i++
____________________________________
while Ture:
    blocks
    if condition:
        break						# come out of while loop
____________________________________
a = 5								# limit in 5 lines
b = 1								# line num, start from 1
while a>0:
	while b<=5:
		print ("*"*b)				# print "*" b times
		b = b+1
		a = a-1
____________________________________
#Digital dice
#importing random function to genterate random number
from random import randint
print("Give lower limit of dice")
a = int(input())					# input 1
print("Give upper limit of dice")
b = int(input())					# input 6
print("type q to Quit or any other key/enter to continue")
while True:
	print(">>> "+str(randint(a,b)))	# randint is generating random num in 1-6
	if input() == 'q':				# if 'q' is entered then come out of loop
		break
```

### 2.3 For

```python
a = [78,98,50,37,45]
for m in a:							# map all ele in list
	print(m)
____________________________________
sum = 0								# init the sum=0
for m in a:							# map all ele in list
	sum = sum+m						# calc the sum value
print(sum)
____________________________________
table_12 = [12,24,36,48,60,72,84,96,108,120]
table_13 = []
z = 1								# delta
for i in table_12:
	table_13.append(i+z)			# from t_12 produce t_13's ele
	z = z+1							# produce a new delta
print(table_13)
```

## 3. File and Directory I/O

### 3.1 File, directory

```python
file = open("new","w")				# opening 'new'(filename) in 'write' mode
print(file)
file.write("This is first line.")	# writing on file, "file.write()"
file.write("This is second line.")
file.close()						# closing the file
____________________________________
file = open("new","r")				# opening 'new'(filename) in 'read' mode
t = file.read()						# reading file, "file.read()"
print(t)
file.close()
____________________________________
file = open("new","r")
t = file.read(2)					# read 2 chr, arg limits num of chr to read
print(t)							# (Th)
t = file.read(999)					# read 999 chr
print(t)							# (is is first line.This is second line.)
file.close()
```

### 3.2 OS module

```python
import os
os.getcwd()							# pwd
os.listdir(path)					# ls
os.chdir(path)						# cd
```

| function            | means       | function              | means            |
| ------------------- | ----------- | --------------------- | ---------------- |
| os.rename(old, new) | mv old new  | os.link(src, dst)     | ln               |
| os.remove(path)     | rm          | os.symlink(src, dst)  | ln -s            |
| os.removedirs(path) | rmdir       | os.readlink(path)     | get path of link |
| os.rmdir(path)      | rm -rf      | os.lchmod(path, mode) | chmod            |
| os.unlink(path)     | delete path |                       |                  |

## 4. Regular expression

```python
#!/usr/bin/python
import re							# import re module
re.match(pattern, string, flag=0)	# match pattern in string(from start)
re.search(pattern, string, flag=0)	# search pattern in string(in full string)
re.sub(pattern, repl, string, count=0, flags=0)	# sub 'pattern' with 'repl', total 'count' times
____________________________________
#!/usr/bin/python
import re
def double(matched):				# define func 'double()'
    value = int(matched.group('value'))
    return str(value * 2)

s = 'A23G4HFD567'
print(re.sub('(?P<value>\d+)', double, s))	# sub 'match pattern' with 'func double()'
```

### 4.1 chr in special

| chr   | means                       | chr    | means                               |
| ----- | --------------------------- | ------ | ----------------------------------- |
| \w    | word                        | \d     | digit, num                          |
| \W    | none word                   | \D     | none digit, num                     |
| \s    | space                       | \b     | match border, 'er\b' -> 'never'     |
| \S    | none space                  | \B     | match none border, 'er/B' -> 'verb' |
| \A, ^ | ahead of string             | \n, \t | next line, tab                      |
| \Z, $ | end of string(current line) |        |                                     |
| \z    | end of string(multi line)   | a\|b   | match 'a' or 'b'                    |

### 4.2 classes

| chr     | means          | chr         | means                |
| ------- | -------------- | ----------- | -------------------- |
| [aeiou] | one of 'aeiou' | [pP]ython   | 'python' or 'Python' |
| [0-9]   | num            | [a-zA-Z0-9] | any 'chr' and 'num'  |
| [a-z]   | chr, lowercase | [^abc]      | none of 'abc'        |
| [A-Z]   | chr, uppercase | [^0-9]      | none of num          |

### 4.3 times

| chr  | times       | chr     | times             |
| ---- | ----------- | ------- | ----------------- |
| re?  | 0, 1 time   | re{n}   | n times           |
| re+  | 1-any times | re{n,}  | n times, at least |
| re*  | 0-any times | re{n,m} | n-m times         |

## 5. Function

```python
def is_even(x):						# define a function, by "def func(arg): block"
	if x%2 == 0:					# condition
		print("even")				# output
	else:
		print("odd")
is_even(2)							# use function
is_even(3)
____________________________________
def sum():							# define a function sum()
	a = int(input("Enter first number >>>"))
	b = int(input("Enter second number >>>"))
	print(a+b)						# function block
sum()
____________________________________
def checkdiv(x,y):
	if x>=y:						# first condition
		if x%y == 0:				# function block
			print(x,"is divisible by",y)
		else:
			print(x,"is not divisible by",y)
	else:
		print("Enter first number greater than or equal to second number")
checkdiv(4,2)
checkdiv(4,3)
checkdiv(2,4)
____________________________________
def is_even(x):						# define a func 'is_even()'
	if x%2 == 0:
		return True					# return result
	else:
		return False
print(is_even(1))					# use the func 'is_even' with arg
print(is_even(2))
____________________________________
def rev(a):							# define a func 'rev()'
	c = []
	i = len(a)-1					# list is start from 0.
	while i>=0:
		c.append(a[i])				# pick ele start from end of list 'c'
		i = i-1						# direction is backword
	return c
z = rev([2,4,6,3,5,2,6,43])			# use the func 'rev' with args
print(z)
____________________________________
def is_even(x):						# define func
	if x%2 == 0:
		return True
	else:
		return False
# div6 function to check divisiblity by 6
def div6(y):						# define func_2
	if is_even(y) and y%3 == 0:		# use 'func' in func_2
		return True
	else:
		return False
____________________________________
def factorial(x):					# define func 'factorial()'
	if x==0 or x==1:
		return 1					# 0!, 1! is 1
	else:
		return x*factorial(x-1)		# clac x!
print(factorial(0))					# 1
print(factorial(1))					# 1
print(factorial(4))					# 24
print(factorial(5))					# 120
```

## 6. Class, object

### 6.1 define class, use object, define method

```python
class Square():
	pass							# none code in block
x = Square()
x.side = 14							# arg is told
print(x.side)
____________________________________
class Square():						# define class
	def perimeter(self,side):		# def a function/method
		return side*4				# how to clac
a = Square()						# use object
print(a.perimeter(14))				# use the func with arg
____________________________________
class Student():					# define class
	def __init__(self,name):		# define init method
		self.name = name
a = Student("Sam")					# use object with arg
print(a.name)
____________________________________
class Rectangle():					# define class
	def __init__(self,l,b):			# define init method
		self.length = l
		self.breadth = b
	def getArea(self):				# define func/method 'getArea()'
		return self.length*self.breadth
	def getPerimeter(self):			# define func/method 'getPerimeter'
		return 2*(self.length+self.breadth)
a = Rectangle(2,4)					# use object with args
print(a.getArea())
print(a.getPerimeter())
```

### 6.2 subclass

```python
class Child():						# define class 'Child'
	def __init__(self,name):		# define init/name method
		self.name = name
class Student(Child):				# define class 'Student', subclass of 'Child()'
	def __init__(self,name,roll):	# define init/name/roll method of subclass
		self.roll = roll
		Child.__init__(self,name)
a = Child("xyz")					# object of 'Child'
print(a.name)
b = Student("abc",12)				# object of 'Student'
print(b.name)
print(b.roll)
____________________________________
class Rectangle():					# define class 'Rectangle'
	def __init__(self,leng,br):		# define init/leng/br method
		self.length = leng
		self.breadth = br
	'''while calling a method in a class python
	automatically passes an instance( object ) of it.
	so we have to pass sef in area i.e. area(self)'''
	def area(self):					# define func/method 'area()'
		'''length and breadth are not globally defined.
		So, we have to access them as self.length'''
		return self.length*self.breadth
class Square(Rectangle):			# define subclass
	def __init__(self,side):
		Rectangle.__init__(self,side,side)
		self.side = side
s = Square(4)						# get a arg
print(s.length)
print(s.breadth)
print(s.side)
print(s.area())						# It appears as nothing is passed but python will pass an instance of class.
```

## 7. Override, exception handle

### 7.1 Overriding

```python
class Rectangle():					# define class
	def __init__(self,length,breadth):	# define init/length/breadth method
		self.length = length
		self.breadth = breadth
	def getArea(self):				# define method
		print(self.length*self.breadth," is area of rectangle")
class Square(Rectangle):			# define subclass
	def __init__(self,side):		# define init/side method
		self.side = side
		Rectangle.__init__(self,side,side)
	def getArea(self):				# define init method
		print(self.side*self.side," is area of square")
s = Square(4)						# use object with arg
r = Rectangle(2,4)					# use object with arg
s.getArea()
r.getArea()
```

### 7.2 Exception handle

```python
try:
    print(5/0)						# if block have error, run 'except block'
except ZeroDivisionError:			# define the error
    print("Division by 0 is not permitted.")	# give the info of error
____________________________________
try:
    a = int(input("hello world"))
    print(a//2)						# a should be a num
except:
    print("Some Error Occurred!")	# if you input 'abc' will get this error
else:
    print("No Errors!")
finally:							# no matter what condition, do the block finally
    print("Either of try or except is executed.")
____________________________________
def divide(a, b):
    try:
        print(a//b)					# define func/method
    except TypeError:				# diff kinds of error
        print('Check the type of arguments.')
    except ZeroDivisionError:
        print('Error: Division by 0')
    except:
        print('Unknown Error')

print("5/\'2\'")
divide(5, '2')						# use the func, test error
print('')							# keep empty, select an error

print("5/0")
divide(5, 0)
print('')

print('5/2')						# use the func, normally
divide(5, 2)
____________________________________
try:
    print(5//0)
except ZeroDivisionError as ex:		# define keyword/label of error info
    print(ex)						# use keyword/label
____________________________________
raise IndexError('Index is out of the range.')	# raise error_type(error_info)
'''
Traceback (most recent call last):	# show the full error info
  File "b.py", line 2, in <module>
    raise IndexError('Index is out of the range.')
IndexError: Index is out of the range.	# if no error_info, just show error_type
'''
```

