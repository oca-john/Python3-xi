#!/usr/bin/python3
# python3.basic.learn.notes

# 1. 基础语法要素
# 赋值方法，直接用“=”赋值
# 数字直接赋值，字串需要引号
counter = 100          # An integer assignment
name    = "John"       # A string
print (counter)
print (name)
a, b = 122, 'hello'     # 同时赋值多个变量
print (a, b)

# 字符串，可用单引号或双引号包含
# 字符串可用[index]进行切片，即提取元素
str = 'Hello World!'
print (str)          # Prints complete string
print (str[0])       # Prints first character of the string
print (str[2:5])     # Prints characters starting from 3rd to 5th
print (str[2:])      # Prints string starting from 3rd character
print (str * 2)      # Prints string two times
print (str + "TEST") # Prints concatenated string

# 列表存储一组变量值
# 列表支持切片操作
list = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
tinylist = [123, 'john']
print (list)          # Prints complete list
print (list[0])       # Prints first element of the list
print (list[1:3])     # Prints elements starting from 2nd till 3rd
print (list[2:])      # Prints elements starting from 3rd element
print (tinylist * 2)  # Prints list two times
print (list + tinylist) # Prints concatenated lists

# 元组存储一组不可变的变量值
# 元组除元素不可变之外，继承列表的大多数性质和操作方法
tuple = ( 'abcd', 786 , 2.23, 'john', 70.2  )
tinytuple = (123, 'john')
print (tuple)           # Prints complete tuple
print (tuple[0])        # Prints first element of the tuple
print (tuple[1:3])      # Prints elements starting from 2nd till 3rd
print (tuple[2:])       # Prints elements starting from 3rd element
print (tinytuple * 2)   # Prints tuple two times
print (tuple + tinytuple) # Prints concatenated tuple

# 字典是无序地存储键值对的变量，类似与哈希
# 字典无需，提取元素通过键值实现，键值具有唯一性
dict = {}
dict['one'] = "This is one"
dict[2]     = "This is two"
tinydict = {'name': 'john','code':6734, 'dept': 'sales'}
print (dict['one'])       # Prints value for 'one' key
print (dict[2])           # Prints value for 2 key
print (tinydict)          # Prints complete dictionary
print (tinydict.keys())   # Prints all the keys
print (tinydict.values()) # Prints all the values

# 算术运算，支持常见的所有运算类型
a = 21
b = 10
c = 0
c = a + b
print ("Line 1 - Value of c is ", c)
c = a - b
print ("Line 2 - Value of c is ", c )
c = a * b
print ("Line 3 - Value of c is ", c)
c = a / b
print ("Line 4 - Value of c is ", c )
c = a % b
print ("Line 5 - Value of c is ", c)
a = 2
b = 3
c = a**b
print ("Line 6 - Value of c is ", c)
a = 10
b = 5
c = a//b
print ("Line 7 - Value of c is ", c)

# 更新列表，列表支持为元素重新赋值，替换之前的值
list = ['physics', 'chemistry', 1997, 2000]
print ("Value available at index 2 : ", list[2])
list[2] = 2001
print ("New value available at index 2 : ", list[2])

# 删除列表，del命令可删除列表的元素或整个列表
list = ['physics', 'chemistry', 1997, 2000]
print (list)
del list[2]     # 删除index为2的元素（即第三个元素）
print ("After deleting value at index 2 : ", list)

# 计算列表长度，len方法可计算列表中元素数目，即列表长度
list1 = ['physics', 'chemistry', 'maths']
print ('length of list1 is:', len(list1))

# max方法，计算列表中的最大值
list1, list2 = ['C++','Java', 'Python'], [456, 700, 200] # 同时赋值两个变量
print ("Max value element : ", max(list1))  # 字串按字母表排大小
print ("Max value element : ", max(list2))

# min方法，计算最小值
list1, list2 = ['C++','Java', 'Python'], [456, 700, 200]
print ("min value element : ", min(list1))
print ("min value element : ", min(list2))

# 比较运算符，比较元素大小，输出为逻辑型
a = 21
b = 10
print(a == b)  # 判断两个元素是否相等，必须用==
print(a != b)
print(a < b)
print(a > b)
print(a <= b)
print(a >= b) # 大于等于包含边界元素

# 基本运算
a = 4
b = 10
c = 3
print(a + b)  # 加法
c += a; print(c) # 加等（先加，后赋值）
c *= a; print(c) # 乘等（先乘，后赋值）
c /= a; print(c) # 除等
c  = 2
c %= a; print(c) # 先取模，后赋值
c **= a; print(c)# 先乘方，后赋值
c //= a; print(c)# 先取余，后赋值

# in操作，表示某变量是某数组中的元素
a = 12
list = [1,23,43,12,54]
print(a in list)    # 判断a变量的值，是否在list列表中存在

====================================================

# 2. 列表操作方法
# append方法，在末尾追加元素
list1 = ['C++', 'Java', 'Python']
list1.append(['C#','C'])    # 多个元素作为整体被追加
print ("updated list : ", list1)

# count方法，统计元素在列表中出现的次数
aList = [123, 'xyz', 'zara', 'abc', 123];
print ("Count for 123 : ", aList.count(123))
print ("Count for zara : ", aList.count('zara'))

# extend方法，追加多个元素
list1 = ['physics', 'chemistry', 'maths']
list2 = ['tst','tst2','tst3']   # 多个元素被逐一追加
list1.extend(list2)
print ('Extended List :', list1)

# index方法，获得元素在列表中的索引号
list1 = ['physics', 'chemistry', 'maths', 'C#']
print ('Index of chemistry:', list1.index('chemistry'))
print ('Index of C#:', list1.index('C#'))

# insert方法，插入元素
list1 = ['physics', 'chemistry', 'maths']
list1.insert(1, 'Biology')  # 在指定索引位置插入元素
print ('Final list : ', list1)

# pop方法，从列表中取出一个元素（根据索引号）
list1 = ['physics', 'Biology', 'chemistry', 'maths']
list1.pop()     # 默认从末尾取出一个元素
print ("list now : ", list1)
list1.pop(1)    # 从指定索引取出一个元素
print ("list now : ", list1)

# remove方法，从列表中删除指定元素（根据元素值）
list1 = ['physics', 'Biology', 'chemistry', 'maths']
list1.remove('Biology')
print ("list now : ", list1)
list1.remove('maths')
print ("list now : ", list1)

# reverse方法，让列表逆序
list1 = ['physics', 'Biology', 'chemistry', 'maths']
list1.reverse()
print ("list now : ", list1)

# sort方法，对列表元素进行排序
list1 = ['physics', 'Biology', 'chemistry', 'maths']
list1.sort()    # 默认按字母表/数字排序
print ("list now : ", list1)

====================================================

# 3. 元组操作方法
# 元组的定义，和用切片的方式取值
tup1 = ('physics', 'chemistry', 1997, 2000)
tup2 = (1, 2, 3, 4, 5, 6, 7 )
print ("tup1[0]: ", tup1[0])
print ("tup2[1:5]: ", tup2[1:5])

# 元组不可以修改元素值，只能基于已有元组创建新元组
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')
# tup1[0] = 100;    # 不支持修改
tup3 = tup1 + tup2  # 创建新元组
print (tup3)

# del操作删除元组
tup = ('physics', 'chemistry', 1997, 2000);
print (tup)
del tup     # 删除后元组将不存在，变为“未定义”状态
print ('after del:', tup)

# len方法与列表中一样，计算元组包含的元素总数
tuple1, tuple2 = (123, 'xyz', 'zara'), (456, 'abc') # 分别赋值
print ("First tuple length : ", len(tuple1))
print ("Second tuple length : ", len(tuple2))

====================================================

# 4. 字典操作方法
# 字典根据键值对用打括号{}创建，键值中间用冒号:表示成对关系
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'};
print(dict)     # 访问所有键值对
print('name:',dict['Name']) # 根据键，查找对应的值

# del方法，可用于删除字典中的一个键值对，或整个字典
# clear方法，用于清空字典内容
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
del dict['Name'] # remove entry with key 'Name'
print(dict)
dict.clear()     # remove all entries in dict
print(dict)
del dict         # delete entire dictionary
print(dict)

# 键名不可重复，否则只输出最后一对
dict = {'Name': 'Zara', 'Age': 7, 'Name': 'Manni'}
print ("dict['Name']: ", dict['Name'])

# len()函数计算数据长度
dict = {'Name': 'Manni', 'Age': 7, 'Class': 'First'}
print ("Length : %d" % len(dict))  # 核心是len(dict)计算字典长度
# %是格式化输出，"%d"是digit占位符，引号外"% len(dict)"是真正函数

# str()函数将数据转为字符
dict = {'Name': 'Manni', 'Age': 7, 'Class': 'First'}
print ("Equivalent String : %s" % str(dict))
# %是格式化输出，"%s"是string占位符，引号外"% str(dict)"是真正函数

# type()函数获得数据类型信息
dict = {'Name': 'Manni', 'Age': 7, 'Class': 'First'}
print ("Variable Type : %s" %  type (dict))

# copy方法可以复制一个字典的内容
dict1 = {'Name': 'Manni', 'Age': 7, 'Class': 'First'}
dict2 = dict1.copy()
print ("New Dictionary : ",dict2)

# get方法，从字典中查找键值对
dict = {'Name': 'Zara', 'Age': 27}
print ("Value : %s" %  dict.get('Age')) # 存在，返回相应的值
print ("Value : %s" %  dict.get('Sex', "NA"))   # 不存在，返回None，可自定义返回值为NA或其他信息

# item方法，提取字典中的键值对
dict = {'Name': 'Zara', 'Age': 7}
print ("Value : %s" %  dict.items())    # 格式化输出%S表示string占位符

# setdefault方法，设置默认值，存在就输出，不存在输出指定的默认值
dict = {'Name': 'Zara', 'Age': 7}
print ("Value : %s" %  dict.setdefault('Age', None)) # 格式化输出%S，string
print ("Value : %s" %  dict.setdefault('Sex', None)) # 不存在时，默认None
print (dict)    # 测试过后，增加了新选项

# update方法，向字典中增加新的键值对
dict = {'Name': 'Zara', 'Age': 7}
dict2 = {'Sex': 'female' }
dict.update(dict2)
print ("updated dict : ", dict)

# values方法，提取字典中的所有值
dict = {'Sex': 'female', 'Age': 7, 'Name': 'Zara'}
dict.values()

====================================================

# 5. Time时间模块方法
# time模块
import time;
print(time.localtime()) # 获得当前时间的详细信息

import time
t = time.localtime()
print ("asctime : ",time.asctime(t))

# calendar模块
import calendar
cal = calendar.month(2020, 6)   # 获得指定月份的日历
print ("Here is the calendar:")
print (cal)

# ctime方法，获得当前时间
import time
print ("ctime : ", time.ctime())

# time模块的sleep方法
import time
print ("Start : %s" % time.ctime())
time.sleep( 5 )         # 暂停指定时间（秒）
print ("End : %s" % time.ctime())

# mktime方法，将以元组形式存储的时间转换为正确的时间表示格式
import time
t = (2015, 12, 31, 10, 39, 45, 1, 48, 0)
t = time.mktime(t)
print (time.strftime("%b %d %Y %H:%M:%S", time.localtime(t)))

====================================================

# 6. 函数及其参数
# def func_name()定义函数，func_name()调用函数
# 定义时要设置形参，调用时要输入实参
def printme(str):   # 定义函数
   print (str)      # 本质是复制了print()函数
   return
printme("This is first strings.") # 调用

# 调用函数时，如果实参是变量，则需要引用变量的真实值作为实参，这就是引用传递
def printme(str):   # 定义函数
   print (str)      # 本质是复制了print()函数
   return
strs = "This is second strings." # 为了和关键字区分，用strs
printme(strs) # 通过引用调用

# 调用时，若已知形参的关键字，可通过给形参赋值来调用函数（关键字参数）
def printme(str):   # 形参关键字为str
   print (str)
   return
printme(str = 'This is third strings.') # 通过给str赋值，直接调用

# 函数可支持多个关键字参数
def printinfo( name, age ):     # 定义函数带有两个关键字
   print ("Name: ", name)
   print ("Age ", age)
   return
printinfo( age = 50, name = "miki" )    # 给关键字赋值，直接调用

# 函数的关键字可以在定义时就指定默认值
def printinfo( name, age = 35 ):    # 定义时指定默认的age值
   print ("Name: ", name)
   print ("Age ", age)
   return
printinfo( age = 50, name = "miki" ) # 调用时更新age的值，则用新的值
printinfo( name = "miki" )      # 调用时不更新age的值，保留默认值

# 函数支持可变参数
def printinfo( arg1, *vartuple ):   # arg1是固定参数，后者是参数元组
   print ("Output is: ")
   print (arg1)
   for var in vartuple:     # var是中间参数，在每个循环中值是变化的（是变量）
      print (var)
   return
printinfo( 10 )
printinfo( 70, 60, 50 )     # 可变参数以元组形式传入，逐个代入

# 匿名函数，不带有显式的def关键字的函数
# 函数名，lambda（用于定义简单的匿名函数），形参：冒号后面接公式（用形参表示）
sum = lambda arg1, arg2: arg1 + arg2
print ("Value of total : ", sum( 10, 20 ))
print ("Value of total : ", sum( 20, 20 ))

# return操作在函数末尾返回一个值
def sum( arg1, arg2 ):
   total = arg1 + arg2
   print ("Inside the function : ", total)  # 函数内直接输出结果
   return total             # 让函数具有return total的功能
total = sum( 10, 20 )       # 如果没有return，则无法输出
#（此处是调用函数，没有真正的加法过程，是否返回结果是根据函数里的定义爱决定的）
print ("Outside the function : ", total )

====================================================

# 7. 全局变量
total = 0       # 全局变量
def sum( arg1, arg2 ):
   total = arg1 + arg2;     # 局部变量，修改sum函数内的total变量
   print ("Inside the function local total: ", total)
   return total
sum( 10, 20 )
print ("Outside the function global total: ", total) # sum函数以外total未修改

====================================================

# 8. 包操作
# import导入包
import torchvision              # 导入包
import torchvision as tv        # 导入包，给包的缩写
from matplotlib import pyplot   # 导入包中的函数
from matplotlib import pyplot as plt    # 导入包中的函数，给函数的缩写
# 通过dir查看python库中的函数
import math
content = dir(math) # math本质是py的lib下的文件夹，可通过dir查看所有子函数
print (content)

var = 100   # 赋值
if ( var == 100 ) : print ("Result is 100") # if 条件：为真时执行的语句
print ("Good bye!")

====================================================

# 9. 文件操作
# open打开文件句柄，w表示写入
# write用于文件写入
fo = open("foo.txt", "w")
fo.write( "Python is a great language.\nYeah its great!!\n") # .write()写入
fo.close()          # .close()关闭句柄

# open打开文件句柄，wb表示以二进制写入，只能写
fo = open("foo.txt", "wb")
print ("Name of the file: ", fo.name)
print ("Closed or not : ", fo.closed)
print ("Opening mode : ", fo.mode)
fo.close()          # .close()关闭句柄

# open打开文件句柄，r+表示读写
# read一次读完，作为一个变量
# readline一次读一行，作为一个变量
# readlines一次读完，作为列表
fo = open("foo.txt", "r+")
str = fo.read(10)   # 读入10行，赋值给str变量
print ("Read String is : ", str)
fo.close()          # .close()关闭句柄

# readlines用于一次读取句柄文件的所有内容
# open打开文件句柄，r+表示读写
fo = open("foo.txt", "r+")
print ("Name of the file: ", fo.name)   # 打印文件名
line = fo.readlines()       # 逐行读取所有
print ("Read Line: %s" % (line))
line = fo.readlines(2)      # 逐行读取2行
print ("Read Line: %s" % (line))
fo.close()          # .close()关闭句柄

# seek重置指针到指定位置，(0,0)表示起始位置
fo = open("foo.txt", "r+")
print ("Name of the file: ", fo.name)
line = fo.readlines()   # 一次读取整个文件（指针到末尾），作为一个字符串
print ("Read Line: %s" % (line))
fo.seek(0, 0)           # 重新设置指针到开始
line = fo.readline()    # 逐行读取文件
print ("Read Line: %s" % (line))
fo.close()

# tell返回当前指针的位置
fo = open("foo.txt", "r+")
print ("Name of the file: ", fo.name)
line = fo.readline()
print ("Read Line: %s" % (line))
pos=fo.tell()           # 查找当前指针位置
print ("current position : ",pos)
fo.close()

====================================================

# 10. OS模块方法
# os模块access方法，访问系统路径下的文件
import os, sys
ret = os.access("/tmp/foo.txt", os.F_OK)    # 测试文件是否存在
print ("F_OK - return value %s"% ret)
ret = os.access("/tmp/foo.txt", os.R_OK)    # 测试文件是否可读
print ("R_OK - return value %s"% ret)
ret = os.access("/tmp/foo.txt", os.W_OK)    # 测试文件是否可写
print ("W_OK - return value %s"% ret)
ret = os.access("/tmp/foo.txt", os.X_OK)    # 测试文件是否可执行
print ("X_OK - return value %s"% ret)

# os模块打开文件、写入、关闭文件
import os, sys
fd = os.open( "foo.txt", os.O_RDWR|os.O_CREAT ) # 用os模块打开文件，后面的参数表示：若存在就以读写方式打开，若不存在就创建
line = "this is test"
b = str.encode(line)    # 指定内容变量，需要转为字节变量encode
os.write(fd, b)         # 将内容写入文件句柄
os.close(fd)            # 关闭文件句柄
print ("Closed the file successfully!!")

os.open()   # 打开文件句柄
# os.fdopen() # 通过fd句柄打开文件
os.read()   # 读取文件
os.write(fd,str)  # 写入文件
os.close()  # 关闭文件句柄
os.chdir()  # cd更改当前目录
# os.fchdir(fd) # 通过fd句柄更改目录，f指文件句柄，chdir=cd
os.mkdir()  # 创建目录
# os.makedirs() # 创建目录-递归
os.getcwd() # pwd输出当前目录
os.lseek(fd,0,0)  # seek重置指针到line，作用与fd句柄，重置到0,0位置

# os.symlink()    # 创建一个软链接
# os.readlink()   # 返回软链接指向的文件路径
import os
src = '/usr/bin/python' # 源地址
dst = '/tmp/python'     # 目标地址
os.symlink(src, dst)    # 创建软链接
print "软链接创建成功"

====================================================

# 11. 测试语句
# Try Else语句
try:
   fh = open("testfile", "w")
   fh.write("This is my test file for exception handling!!")
except IOError:         # 如果try部分引发IOError异常，则执行
   print ("Error: can\'t find file or read data")
else:                   # 没发生异常，则执行
   print ("Written content in the file successfully")
   fh.close()

# Finally语句
try:
   fh = open("testfile", "w")
   fh.write("This is my test file for exception handling!!")
finally:                # 不论try语句是否通过，都将执行
   print ("Error: can\'t find file or read data")
   fh.close()

====================================================

# 12. 面向对象、类、方法
# 暂时无法加注释
class Employee:
   'Common base class for all employees'
   empCount = 0

   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1

   def displayCount(self):
     print ("Total Employee %d" % Employee.empCount)

   def displayEmployee(self):
      print ("Name : ", self.name,  ", Salary: ", self.salary)
emp1 = Employee("Zara", 2000)
emp2 = Employee("Manni", 5000)
print ("Employee.__doc__:", Employee.__doc__)
print ("Employee.__name__:", Employee.__name__)
print ("Employee.__module__:", Employee.__module__)
print ("Employee.__bases__:", Employee.__bases__)
print ("Employee.__dict__:", Employee.__dict__ )
---------------------------------------
class Point:
   def __init__( self, x=0, y=0):
      self.x = x
      self.y = y
   def __del__(self):
      class_name = self.__class__.__name__
      print (class_name, "destroyed")

pt1 = Point()
pt2 = pt1
pt3 = pt1
print (id(pt1), id(pt2), id(pt3));   # prints the ids of the obejcts
del pt1
del pt2
del pt3
---------------------------------------
class Parent:        # define parent class
   parentAttr = 100
   def __init__(self):
      print ("Calling parent constructor")

   def parentMethod(self):
      print ('Calling parent method')

   def setAttr(self, attr):
      Parent.parentAttr = attr

   def getAttr(self):
      print ("Parent attribute :", Parent.parentAttr)

class Child(Parent): # define child class
   def __init__(self):
      print ("Calling child constructor")

   def childMethod(self):
      print ('Calling child method')
c = Child()          # instance of child
c.childMethod()      # child calls its method
c.parentMethod()     # calls parent's method
c.setAttr(200)       # again call parent's method
c.getAttr()          # again call parent's method
---------------------------------------
class Parent:        # define parent class
   def myMethod(self):
      print ('Calling parent method')

class Child(Parent): # define child class
   def myMethod(self):
      print ('Calling child method')

c = Child()          # instance of child
c.myMethod()         # child calls overridden method
---------------------------------------
class Vector:
   def __init__(self, a, b):
      self.a = a
      self.b = b

   def __str__(self):
      return 'Vector (%d, %d)' % (self.a, self.b)

   def __add__(self,other):
      return Vector(self.a + other.a, self.b + other.b)
v1 = Vector(2,10)
v2 = Vector(5,-2)
print (v1 + v2)
---------------------------------------
class JustCounter:
   __secretCount = 0

   def count(self):
      self.__secretCount += 1
      print (self.__secretCount)

counter = JustCounter()
counter.count()
counter.count()
print (counter.__secretCount)

====================================================

# 13. 正则表达式
# re正则表达式，match方法用于正则匹配
import re       # 载入re包
line = "Cats are smarter than dogs"     # 匹配源文件
matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)
if matchObj:
   print ("matchObj.group() : ", matchObj.group())  # 捕获完整匹配
   print ("matchObj.group(1) : ", matchObj.group(1)) # 捕获第一个()
   print ("matchObj.group(2) : ", matchObj.group(2)) # 捕获第二个()
else:
   print ("No match!!")

# search方法用于正则匹配
import re
line = "Cats are smarter than dogs";
searchObj = re.search( r'(.*) are (.*?) .*', line, re.M|re.I)
if searchObj:
   print ("searchObj.group() : ", searchObj.group())
   print ("searchObj.group(1) : ", searchObj.group(1))
   print ("searchObj.group(2) : ", searchObj.group(2))
else:
   print ("Nothing found!!")

# match和search的比较
import re
line = "Cats are smarter than dogs";
matchObj = re.match( r'dogs', line, re.M|re.I)  # 从开头开始匹配
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
else:
   print ("No match!!")
searchObj = re.search( r'dogs', line, re.M|re.I)# 在整个文本中搜索匹配
if searchObj:
   print ("search --> searchObj.group() : ", searchObj.group())
else:
   print ("Nothing found!!")

re.match('sub', 'substring')    # 从开头匹配
re.search('str', 'substring')   # 在文本中搜索匹配项
re.findall('a', 'a aa ab ac')   # 返回所有符合a的匹配结果，放入列表
re.split('[ab]', 'abcd')        # 先按a分割文本，再按b分割子文本
re.sub('\d', 'H', 'eva3egon4yuan4',1)   # 将文本中的\d替换为H，替换1次
re.subn('\d', 'H', 'eva3egon4yuan4')    # 将文本中的\d替换为H，返回替换结果和替换次数

====================================================

# 14. 数字操作
abs(-45)            # abs()返回参数的绝对值（内置函数）
import math
math.fabs(-45)      # 返回参数的绝对值（math模块中的）
math.ceil(-4.17)    # 返回上进整数，-4
math.ceil(10.12)    # 返回上进整数，11
math.floor(-4.17)   # 返回下舍整数，-5
math.floor(10.12)   # 返回下舍整数，10
math.exp(4)         # 返回e的()次方
math.log(100)       # 返回以e为底，参数的对数
math.log(math.e)    # math.e调用e的值，log结果为1
math.log(8,2)       # 返回以2为底，参数的对数
math.log10(10000)   # 返回以10为底，参数的对数

max(1,4,45)         # 返回参数中最大值
min(1,4,45)         # 返回参数中最小值
import math
math.modf(2.12)     # 返回参数的小数和整数部分（浮点）
math.pow(3,2)       # 返回3的2次方
math.sqrt(100)      # 返回参数的开方

import random
random.choice(range(32))    # 从区间随机选一个数
random.choice([2,4,5,3,8])  # 从列表中随机选一个数
random.choice('helloyou')   # 从文本中随机选一个字符
random.randrange(0,100,5)   # 在0-100之间，以5为步长，生成数组，并随机返回一个数
random.random()     # 返回一个(0,1)返回内的实数
random.seed(2)      # 设置seed，让random每次生成相同随机数
random.random()     # 返回一个(0,1)范围内的实数
random.uniform(5,10)# 返回一个(5,10)范围内的浮点数

import random
list = [20, 16, 10, 5]  # 定义列表
random.shuffle(list)    # 随机重排列表
print(list)

====================================================

# 15. 字串操作
# Python 中，有 2 种常用的字符串类型，分别为 str 和 bytes 类型，其中 str 用来表示 Unicode 字符，bytes 用来表示二进制数据。str 类型和 bytes 类型之间就需要使用 encode() 和 decode() 方法进行转换。

# 格式化输出字串
# %s是string占位符，%d是digit占位符，%(这是里参数)
print ("My name is %s and weight is %d kg!" %('Zara', 21))

# ''' 或 """ 成对的三引号用于多行文本（保留换行的格式）
mult_str = '''this is a long string
which contains multi lines
in it. just have a try.'''
print (mult_str)

# 字符转义，显示原字符串
print ('C:\\nowhere')   # 路径中的\被转义后显示一个
print (r'C:\\nowhere')  # r表达式用于避免任何转义

# str.capitalize将字符串转为首字母大写
# str.lower()全小写
# str.upper()全大写

print ("str.capitalize() : ", str.capitalize())
str.capitalize() :  This is string example....wow!!!
# str.count用于计算'子字串在起点和终点间出现的次数'
str = "this is string example....wow!!!";
sub = "i";
print ("str.count(sub, 4, 40) : ", str.count(sub, 4, 40))

# find()方法检测字串中是否包含子字串str（返回第一个）
# rfind()方法检测字串中是否包含子字串str（返回最后一个）
str1 = "Runoob example....wow!!!"
str2 = "exam";
print (str1.find(str2))
print (str1.find(str2, 5))
print (str1.find(str2, 10))

# endswith()字串是否以指定字符结尾
str.endswith('endstr', start, end)
# startswith()字串是否以指定字符开始
str.startswith('startsstr', start, end)
# isalnum()字串是否由字母(alpha)和数字(num)组成
# isdigit()字串是否只由数字(digit)组成
# isalpha()字串是否只由字母(alpha)组成
# isdecimal()字串是否只包含十进制字符
# islower()是否全小写
# isupper()是否全大写
# isspace()是否只有空格

# split方法用于分割字串
str = "this is string example....wow!!!"
print (str.split( ))        # 默认用空格分割字串
print (str.split('i',1))    # 
print (str.split('w'))

# join方法用于合并字符串，并加入指定分隔符
s = "-"                 # 指定分隔符
seq = ("a", "b", "c")   # 指定序列文件
print (s.join(seq))     # 用s分隔符连接seq序列中的元素

# len()方法用于计算字串长度（即包含多少个字符）
str = "this is string example....wow!!!"
print ("Length of the string: ", len(str))

# replace()方法用于在字串中'替换一部分子字串'
str = "this is string example....wow!!! this is really string"
print (str.replace("is", "was"))    # 默认替换全部匹配项
print (str.replace("is", "was", 3)) # 替换不超过3次（共有4次）

# translate()方法用于按照一定的映射关系执行多次替换，即翻译
intab = "aeiou"     # 源字符集
outtab = "12345"    # 目标字符集
trantab = str.maketrans(intab, outtab)  # 制作翻译表
str = "this is string example....wow!!!"
print (str.translate(trantab))          # 执行翻译并输出
