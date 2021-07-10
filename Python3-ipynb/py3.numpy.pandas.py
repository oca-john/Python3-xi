# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Numpy & Pandas Learning Note
# - VScode中, `ctrl + /` 快速注释代码
# ## 1. numpy 操作数组对象和基本计算

# %%
# 生成数组，指定维数和数据类型

import numpy as np 
arr = np.array([[1,2,3],[5,6,7]], ndmin=3, dtype=float)         # 嵌套[]方式创建，嵌套关系需要正确
arr2 = np.array(np.mat('1,2,3;4,5,6'), ndmin=3, dtype=float)    # np.mat方式创建，指明分组即可
print('数组是:\n', arr)
print('数组2是:\n', arr2)
print('数据类型分别是:\n', arr.dtype, arr2.dtype)
arr_comp = np.array([1, 2, 3], dtype = complex)    # 创建复数数组
print('复数数组是:\n', arr_comp)


# %%
# 生成数据，索引筛选

arr3 = np.full((3,3),2) # 创建3*3的全2矩阵（矩阵形状和数值，两个参数）
print('数组3是:\n', arr3)
arr4 = np.random.random((4,3))
print('数组4是:\n', arr4)
print('数组4切片:\n', arr4[0:,1:2])     # 0:打印所有行，1:2打印第2行（大于等于2，小于3）
print('数组4切片:\n', arr4[-1:,1:2])    # -1打印末行，1:2同上


# %%
# numpy数组属性
# ndarray.shape函数获取数组的维度，也可用于调整数组大小

import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print('原数组维度是：', a.shape)     # 获得数组形状（维度）
a = np.array([[1,2,3],[4,5,6]])
a.shape = (3,2)     # 向数组形状赋值（定义维度）
print('新的数组是：\n', a)
b = a.reshape(2,3)
print('复制的数组是：\n', b)


# %%
# arrange生成数组, ndim获得维度

import numpy as np
a = np.arange(24) # 生成24以内的数据集
b = np.arange(12).reshape(3,4)  # 生成序列数据后，直接reshape形状构成目标数组
print (a)
print(b.ndim)   # 获得数组的维度
print(b.size)
print(b.shape)


# %%
# zeros生成全零张量，ones全一张量

import numpy as np
x = np.zeros(5)
print(x)
x = np.ones(5)    # 生成全一张量，一维5个数
print(x)
x = np.ones([2,4], dtype=int) # 指定生成张量的维度，数据类型
print(x)


# %%
# asarray将数组（列表或元组）解释为张量

import numpy as np
x = [1,2,3]  # 此处使用(1,2,3)元组也可以，转张量后一样
a = np.asarray(x, dtype = float)
print (a)


# %%
# arange给定范围内均匀间隔值的ndarray对象
# linspace在给定范围内生成指定数量的数

import numpy as np
x = np.arange(5)       # 生成5以内的数（0-5）
print (x)
x = np.arange(10,20,2) # 在（10-20）之内，以2为步长
print (x)
x = np.linspace(10,20,5)# 在（10-20）之内，等分生成5个数
print (x)
x = np.linspace(10,20, 5, endpoint = False)# 不包含终点而等分
print (x)


# %%
# logspace在给定范围内生成指定底数的对数对应的指数（底数默认10）

a = np.logspace(1.0, 2.0, num = 10)
print (a)  # 底数为10,将对数1到2等分为10份，返回其指数
a = np.logspace(1,10,num = 10, base = 2)
print (a)  # 底数为2,将对数1到10等分为10份，返回其指数


# %%
# 条件筛选切片
print('数组4筛选切片:\n', arr4[np.arange(4),1:3])   # 生成0-4的数列，用其选取0-4行，手动选择index为1和2的列
print('数组4条件筛选:\n', arr4 > 0.5)       # 逐个元素判断大小，输出布尔值
print('数组4条件筛选:\n', arr4[arr4 > 0.5]) # 逐个元素判断大小，输出符合条件的值


# %%
# slice方法对张量进行切片

import numpy as np
a = np.arange(10) # 生成10以内的数
s = slice(2,7,2)  # 切片取出2到7之间，以2为步长的数
print(a[s])       # 调用和python一样，var[slice]
print(a[3])       # 切片取index为3的子项
print(a[4:])      # 切片取index为4以后的所有子项
print(a[2:8])     # 切片取index为2到8之间的所有子项


# %%
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) # 生成二维数组
print (a)     # 输出数组a
print ('[1:]后的子项为:\n', a[1:]) # 输出数组a的后两项（不输出index为0的项）


# %%
# 基本数学运算

# 加减乘除
arr5 = np.array(np.arange(8))
arr5 = arr5.reshape(2,4)
print('数组5:\n', arr5)
arr6 = np.array([[4,8,7,6],[4,7,5,3]])
arr6 = arr6.reshape(4,2)
print('数组6:\n', arr6)

# 计算
# print('加法:\n', np.add(arr5, arr6))         # 等于 arr5+arr6
# print('减法:\n', np.subtract(arr5, arr6))    # 等于 arr5-arr6
# print('乘法:\n', np.multiply(arr5, arr6))    # 等于 arr5*arr6
# print('除法:\n', np.divide(arr5, arr6))      # 等于 arr5/arr6
# print('平方:\n', np.square(arr5))            # 平方
# print('开方:\n', np.sqrt(arr5))              # 开方

# 矩阵计算
print('数组5和6做矩阵乘法:\n', np.dot(arr5,arr6))        # 矩阵相乘，arr1的列数要等于arr2的行数
print('数组5求和:\n', np.sum(arr5, axis=0))     # 对列操作，矩阵按列求和（按行是1）
print('数组5求平均:\n', np.mean(arr5))


# %%
# 随机数，复制排布矩阵

ran = np.random.uniform(1,199)
print('随机数:\n', ran)
arr7 = np.array([[1,2,3],[4,5,6]])
print('数组7:\n', arr7)
print('数组7赋值填充:\n', np.tile(arr7, (2,3)))     # 复制并生成新的矩阵

# 打印索引，转置
arr7 = np.array([[1,2,3],[4,5,6]])
# print(arr7.argsort())           # 以当前矩阵形式，打印升序排列后的索引号
print('数组7转置:\n', arr7.T)                   # 大写T打印转置矩阵


# %%
# 高级索引，对多维数组进行直接索引，分别指定各个维度的调取范围

import numpy as np    # 下面生成一个4×3的数组
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]])
print(x)
slc = x[1:4,1:3]      # 一维取1:4，二维取1:3
print('切片1:\n', slc)
slc2 = x[1:4,[1,2]]   # 一维取1:4，二维取[1]
print('切片2:\n', slc2)


# %%
# 数据块分割

import numpy as np 
a = np.arange(12).reshape(3,4)
print('a的原始值：\n', a)
print('横向3分：\n', np.split(a, 3, axis=0))    # 横向分3个数据块
print('横向3分：\n', np.vsplit(a, 3))
print('纵向2分：\n', np.split(a, 2, axis=1))    # 纵向分2个数据块
print('纵向2分：\n', np.hsplit(a, 2))
print('非等份分割：\n', np.array_split(a, 3, axis=1)) # 非等份分割


# %%
# numpy中变量数据的关联（torch中也是）
# 使用`.copy`防止关联变化

import numpy as np 
a = np.arange(24)
b = a
print('第1组数据：\n', a, '\n', b)   # 复制后a和b一样

b[3]=34
print('第2组数据：\n', a, '\n', b)   # 修改b后，a也改变

c = a.copy()        # 用`.copy`的方式复制变量
c[3]=333
print('第3组数据：\n', a, '\n', c)   # 


# %%
# 广播机制
# 需要形状相同/某一维等宽/某一维宽度为1/单个元素

arr7 = np.array([[1,2,3],[4,5,6]])
arr8 = np.array([[3],[4]])
print('矩阵处理的广播机制:\n', arr7+arr8)                # 处理不同形状矩阵/数组时，自动填充至较大矩阵的形状

# %% [markdown]
# ## 2. Pandas 操作和筛选序列数据与数据框

# %%
# 生成数据，Series本身的函数

import pandas as pd 
s = pd.Series(list('ABCDE'))
print(s)
# print(s.str.lower())    # 转小写
# print(s.str.upper())
# print(s.str.len())      # 计算字串长度
print(s.str.split(' '))         # 字串工具，切片
print(s.str.replace('A', 'H'))  # 字串工具，替换
# print(s.str.extract('(\d)'))    # 字串工具，正则式，提取字串中带有数字的元素，\d被小括号包裹，可以返回输出
# print(s.str.extract('[ab](\d)'))# 字串工具，正则式，提取a或b开头且带有数字的元素，\d被小括号包裹，可以返回输出
# print(s.str.extract('([ab])(\d)'))  # [ab]和\d分别被小括号包裹，将分别返回输出（2列返回值）
print(s.str.extract('(?P<string>[ab])(?P<number>\d)'))  # 对2列返回值的列标题进行命名，使用"?P<str>"定义列标题


# %%
# 生成序列数据或数据框

import numpy as np 
import pandas as pd

s = pd.Series([1,2,4,5, np.nan, 34,23])     # pandas中最基本的“序列”
print('序列数据：\n', s)

dates = pd.date_range('20190302', periods=6)    # 生成日期序列数据
print('日期数据：\n', dates)

# 构建DataFrame数据框，随机生成数据内容，用dates数据做为行索引，自定义标签作为列标题
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a', 'b', 'c', 'd'])
print('数据库1：\n', df)

# 生成np.array的方式生成数据框，前面用pandas的DataFrame处理过就行
df1 = pd.DataFrame(np.arange(12).reshape(3,4))
print('数据库2：\n', df1)

# 打印行名（索引）和列名
print('行名/索引：\n', df.index)
print('列名：\n', df.columns)
print('值信息：\n', df.values)
print('基本描述：\n', df.describe())

# 按照值进行排序
print('值排序：\n', df.sort_values(by='b', ascending=True)) # 以b列，按升序排序


# %%
s = pd.Series(['a', 'aB', 'ca', 'Da'])
print(s)
rule = r'[a-z]'             # 引号内为正则式规则，匹配小写字母
# print(s.str.contains(rule)) # 字串工具，正则式，判断是否包含符合rule规则的内容
# print(s.str.startswith('a'))
print(s.str.contains('^a')) # 判断是否以a开头
# print(s.str.endswith('a'))
print(s.str.contains('a$')) # 判断是否以a结尾


# %%
# 选择数据(先构造数据)

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape(6,4), index=dates, columns=['A','B','C','D'])
print('构造数据：\n', df)

# 选取
print(df['A'], '\n', df.C)      # 按列名选取数据，两种方式（文本标签/列对象）
print(df[0:3], '\n', df['20130102':'20130104'])

# loc选取（用标签进行数据筛选）
print(df.loc['20130105'])       # 选取index为指定值的行的数据
print(df.loc[:, ['A','B']])     # :表示选择所有行，[]内容表示选择列名为A或B的列
print(df.loc['20130105', ['A','B']])    # 指定行index和列名

# iloc选取（用位置的index）
print('用Index选择：\n', df.iloc[[1,2,3], 2:4])

# ix选取（用标签和index混合筛选数据）   # 未成功使用ix
# print(df.ix[:3, ['A','C']])

# 逻辑选取
print('A列数值大于16的行:\n', df[df.A > 16])

# 选取数据的方式可以用于数据的赋值，如：
# df.loc['20130105', ['B']] = 66
df.iloc[2,2] = 222
print('新数据：\n', df)


# %%
# 读取写入文件

import pandas as pd
# data = pd.read_excel('/path/to/your/excel_file.xlsx', sheet_name='sheet2')  # 可指定工作表名
# data = pd.read_csv('/path/to/your/file.csv')    # pd调用read函数，文件是参数
# data.to_csv('/saving/path/of/file.csv')         # 对象的to方法，输出文件是参数
# data.to_pickle('/saving/path/of/file.pickle')

# pl = data.plot(kind='scatter', x='xlabel', y='ylabel').get_figure() # pandas画图
# pl.savefig('/path/to/fig.jpg')

dates = pd.date_range('20210401', periods=6)
# print(dates)
df = pd.DataFrame(np.random.rand(6,4), index=dates, columns=list('ABCD'))
# print(df)
pl = df.plot(kind='scatter', x='A', y='B').get_figure()     # 不保存则直接显示在交互环境中


# %%
# 生成索引和数据框，数据框切片

dates = pd.date_range('20210401', periods=6)
# print(dates)
df = pd.DataFrame(np.random.rand(6,4), index=dates, columns=list('ABCD'))
# print(df)
print(df['20210401':'20210403'])    # 筛选切片
print(df.loc['20210401':'20210403', ['A', 'C']])    # .loc指定列筛选
# print(df.at[dates[0], 'A'])         # 访问指定元素
print(df.head(2))                   # 查看前2行
# print(df.tail(2))
# print(df.index)                     # 打印索引
# print(df.columns)                   # 打印列名
# print(df.values)                    # 打印值
print(df.describe())                # 描述信息，注意括号
# print(df.T)                         # 数据框转置


# %%
# 数据筛选

dates = pd.date_range('20210401', periods=6)
# print(dates)
df = pd.DataFrame(np.random.rand(6,4), index=dates, columns=list('ABCD'))
# print(df)
print(df.D)
print(df[df.D > 0.5])               # 将D列数据大于0.5作为筛选条件，打印符合条件的所有行
print(df[df.D > 0.5][['A','C']])    # 筛选并打印指定列，注意列选择时，使用两层[]


# %%
# pandas拼接数据

import numpy as np
import pandas as pd

# 构造数据框
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])

# .concat方法拼接数据
res = pd.concat([df1, df2, df3], axis=1)    # 合并三组数据，轴向是横向
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True) # 纵向合并，忽略原标签
print(res)
# concat方法的join参数，inner合并交集部分,outer以并集合并
res = pd.concat([df1,df2], join='outer')    # 当各数据框结构不同时有效
print(res)


# %%
# append追加数据

df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])

# 追加等宽的数据框
apd = df2.append(df3, ignore_index=True)
print(apd)

# 追加等宽的Series数据
sr1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
aps = df2.append(sr1, ignore_index=True)
print(aps)


# %%
# merge函数合并数

import pandas as pd 

# 构造数据
left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                    'A':['A0','A1','A2','A3'],
                    'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                    'C':['C0','C1','C2','C3'],
                    'D':['D0','D1','D2','D3']})
print(left, '\n', right)

# merge合并（唯一关键字合并）
res = pd.merge(left, right, on='key')
print(res)


# %%
# 多个关键字合并

import pandas as pd

# 构造数据
left = pd.DataFrame({'key1':['K0','K0','K1','K1'],
                    'key2':['K0','K1','K0','K1'],
                    'A':['A0','A1','A2','A3'],
                    'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                    'key2':['K0','K0','K1','K1'],
                    'C':['C0','C1','C2','C3'],
                    'D':['D0','D1','D2','D3']})
print(left, '\n', right)

# 合并数据（两个关键字）
res = pd.merge(left, right, on=['key1', 'key2'])    # 默认how=inner
print(res)
res = pd.merge(left, right, on=['key1', 'key2'], how='outer')   # 并集方式合并
print(res)
res = pd.merge(left, right, on=['key1', 'key2'], how='left')    # 以left的key为准
print(res)                          # 可以加入indicator参数，结果中显示合并策略

# by index 合并
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')


# %%



