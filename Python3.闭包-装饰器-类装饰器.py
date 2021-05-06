# 闭包的概念：内部函数引用外部函数的变量，即在函数内部存在一个闭包（内部函数）
def outer(x):			# 传入 x（外部）
    def inner(y):		# 传入 y（内部）
        return x + y 	# 同时调用内部的y和外部的 x，返回 x+y 的结果
    return inner 		# x+y 的结果输出到外部函数，是以内部函数名 inner 的形式出现的，
    					# 返回 inner 即可。

print(outer(6)(5))
-----------------------------
>>>11


# 装饰器的概念：本质也是闭包，是闭包的应用，是用于扩展原来函数功能的函数，其返回值也是一个函数，
# 可在不修改原函数的前提下给函数增加新功能。使用时，需要在函数前加上 @demo 调用。
def debug(func):		# 传入对象函数
    def wrapper():		# 可增加参数（此处无）
        print("[DEBUG]: enter {}()".format(func.__name__))	# 继承外部的 func 并格式化输出
        return func()	# 直接返回 func()
    return wrapper 		# 内部函数以函数名 wrapper 形式返回给外部 debug 函数

@debug 					# 调用 debug 函数作为装饰器修饰后面的 hello()，
						# hello() 将作为 func 参数传入装饰器。
def hello():			# 原 hello() 函数不用修改代码
    print("hello")

hello()					# 调用 hello() 函数
-----------------------------
>>>[DEBUG]: enter hello()
>>>hello


# 带参数的装饰器
def logging(level):						# 接受 INFO 参数并在最外层
    def outwrapper(func):				# 实际接收 func 函数的函数
        def wrapper(*args, **kwargs):	# 内部支持接收参数列表
            print("[{0}]: enter {1}()".format(level, func.__name__))	# 格式化输出
            return func(*args, **kwargs)# 返回所有参数作为内部函数的输出
        return wrapper 					# 逐层返回输出
    return outwrapper

@logging(level="INFO")	# 调用 logging 装饰器，修饰后面的 hello()
def hello(a, b, c):
    print(a, b, c)

hello("hello,","good","morning")
-----------------------------
>>>[INFO]: enter hello()
>>>hello, good morning


# 类装饰器，将对函数的修饰扩展到对函数类的修饰，被修饰的函数被看做对象
class logging(object):
    def __init__(self, func):				# 初始化方法，接收了函数作为参数
        self.func = func
        									# 使用类方法中的 call 方法来直接调用装饰器
    def __call__(self, *args, **kwargs):	# 接收函数和参数，作为参数列表
        print("[DEBUG]: enter {}()".format(self.func.__name__))
        return self.func(*args, **kwargs)	# 返回所有参数作为输出

@logging 				# 调用 logging 类装饰器，修饰后面的 hello 类（被看做对象）
def hello(a, b, c):
    print(a, b, c)

hello("hello,","good","morning")
-----------------------------
>>>[DEBUG]: enter hello()
>>>hello, good morning


# 类装饰器也可以带参数
class logging(object):
    def __init__(self, level):			# 初始化方法，接收了外部参数
        self.level = level

    def __call__(self, func):			# 接收参数和函数，作为参数列表
        def wrapper(*args, **kwargs):	# 内部函数再接收参数列表
            print("[{0}]: enter {1}()".format(self.level, func.__name__))
            return func(*args, **kwargs)# 返回所有参数作为输出
        return wrapper

@logging(level="TEST")					# 装饰器带了参数 level
def hello(a, b, c):
    print(a, b, c)

hello("hello,","good","morning")
-----------------------------
>>>[TEST]: enter hello()
>>>hello, good morning

