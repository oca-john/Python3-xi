# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Matplotlib 学习笔记
# %% [markdown]
# ## 3. Matplotlib 可视化图表
# - VScode中, `ctrl + /` 快速注释代码

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.Series(np.random.randn(1000), index=np.arange(1000))  # 随机生成数据
data = data.cumsum()        # 累加数据并赋值
data.plot()
# 其他绘图方法：bar, hist, box, kde, area, scatter, hexbin, pie


# %%
data = pd.DataFrame(np.random.randn(1000,4), 
                    index=np.arange(1000),
                    columns=list('ABCD'))
data = data.cumsum()
data.plot()


# %%
data = pd.DataFrame(np.random.randn(1000,4), 
                    index=np.arange(1000),
                    columns=list('ABCD'))
data = data.cumsum()
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class 2', ax=ax)


# %%
import matplotlib.pyplot as plt
import numpy as np 
# 绘制二次函数的图像
x = np.linspace(-1, 1, 100)     # 线性方程生成数据
y = x**2                        # 指定y的计算公式，展示函数图形
plt.plot(x,y)
plt.show()  # vscode中可以省略该步骤


# %%
# 用多个面板展示图形（每个plot，用一个figure包含）
import matplotlib.pyplot as plt
import numpy as np 
x = np.linspace(-1, 1, 100)
y1 = 2*x+1
y2 = x**2
plt.figure()                        # 依次绘制图形
plt.plot(x,y1)
plt.figure(num=3, figsize=(5,3))    # 指定面板编号，面板宽和高
plt.plot(x,y2)


# %%
# 同一面板展示多个图形（一个figure，多次plot）
# 线条属性设置
import matplotlib.pyplot as plt
import numpy as np 
x = np.linspace(-1, 1, 100)
y1 = 2*x+1
y2 = x**2
plt.figure()
plt.plot(x,y2)
plt.plot(x,y1, color='red', linewidth=1.0, linestyle='--')
# 指定图形的颜色，线宽，线形

# 坐标轴设置
plt.figure()
plt.plot(x,y2)
plt.plot(x,y1, color='red', linewidth=1.0, linestyle='--')
# 坐标轴范围 lim
plt.xlim(-1,0.9)    # 莫烦教程中有两层括号，推测是旧版本Python只接受标准元组作为参数
plt.ylim(-2,2.1)
# 周标签 label
plt.xlabel('X zuobiao zhou')
plt.ylabel('Y zuobiao zhou')
# 刻度标签 ticks
plt.xticks(np.linspace(-1,0.9, 6))
plt.yticks([-2, -1, 0, 1, 2],       # 自定义数值对应的描述性标签
            ['v.bad','bad','soso','good','v.good'])


# %%
import matplotlib.pyplot as plt
import numpy as np 
x = np.linspace(-1, 1, 100)
y1 = 2*x+1
y2 = x**2
plt.figure()
plt.plot(x,y2, label='leg1')    # 设置图例要先指定不同图对应的label
plt.plot(x,y1, color='red', linewidth=1.0, linestyle='--', label='leg2')
# plt.legend()  # 默认方式
plt.legend(loc='best')

plt.figure()
l1, = plt.plot(x,y2, label='leg1')  # 以handles方式注意`l1,`对象后面的逗号
l2, = plt.plot(x,y1, color='red', linewidth=1.0, linestyle='--', label='leg2')
# plt.legend()  # 默认方式
plt.legend(handles=[l1,l2,], labels=['lineA','lineB'], loc='best')


# %%
# 散点图
# plt.scatter(X,Y,s=size, c=color, alpha=toumingdu)
import matplotlib.pyplot as plt
import numpy as np

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X)     # for color value
plt.scatter(X,Y, s=75,c=T, alpha=0.5)   # s大小，c颜色，marker标记形状
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)


# %%
# 柱状图
# plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')


# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax = Axes3D(fig)
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
X,Y = np.meshgrid(X,Y)
R = np.sqrt(X**2+Y**2)
Z = np.sin(R)
ax.plot_surface(X,Y,Z, rstride=3, cstride=3, cmap=plt.get_cmap('rainbow'))


# %%
# 画布的不规则划分
# 方法一：subplot子图编号手动排布
# import matplotlib.pyplot as plt 
# plt.figure()    # 绘制画布
# plt.subplot(2,1,1)      # 画布分为2行，第一行分为1列（图编号是1，实际占用了1和2）
# plt.plot([0,1],[0,1])
# plt.subplot(2,2,3)      # 2行，第而行分为2列（图编号从3继续）
# plt.plot([0,1],[0,1])
# plt.subplot(2,2,4)      # 2行，第而行分为2列，建立第二个子画布
# plt.plot([0,1],[0,1])

# 方法二：subplot2grid
# plt.figure()
                # 2*2分割画布，ax1从0,0位置起笔，占用2个横向单元1个纵向单元
# ax1 = plt.subplot2grid((2,2), (0,0), colspan=2, rowspan=1)
# ax1.plot([1,2],[1,2])   # 绘制数据
# ax1.set_title('title of ax1')   # 子图设置属性的方法是`.set_xxx`，与plt.xxx不同
# ax2 = plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)  # 1,0表示左下角
# ax2.plot([1,2],[1,2]) 
# ax3 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
# ax3.plot([1,2],[1,2]) 

# 方法三：gridspec
import matplotlib.gridspec as gridspec
plt.figure()
gs = gridspec.GridSpec(2,2) # 划分网格
ax1 = plt.subplot(gs[0,:])  # 用网格的索引位置，标记图像占位
ax1.plot([1,2],[1,2]) 
ax2 = plt.subplot(gs[1,0])
ax2.plot([1,2],[1,2]) 
ax3 = plt.subplot(gs[1,1])
ax3.plot([1,2],[1,2]) 


# %%
# 次坐标轴绘制
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y1 = 0.05*x**2
y2 = -1*y1

fig,ax1 = plt.subplots()# .subplots函数一次输出画布和ax1对象
ax2 = ax1.twinx()       # 坐标轴镜像翻转
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data',color='g')
ax2.set_ylabel('Y2 data',color='b')


# %%
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation

# fig, ax = plt.subplots()
# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))

# def animate(i):
#     line.set_ydata(np.sin(x_i/10))
#     return line,

# def init():
#     line.set_ydata(np.sin(x))
#     return line,

# ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init, interval=20)

# plt.show()  # vscode无法显示动态效果


# %%



