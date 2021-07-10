# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SciKitLearn 机器学习库
# - VScode中, `ctrl + /` 快速注释代码

# %%
# Sklearn 通用的学习模式
# 案例1. 本例鸢尾花数据集，使用KNN模块实现分类

import numpy as np 
from sklearn import datasets
# from sklearn.cross_validation import train_test_split # cross_validation包早已不再使用，功能划入model_selection模块中
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()     # 加载鸢尾花数据集
iris_X = iris.data              # 属性存入X变量，作为特征向量集合
iris_y = iris.target            # 标签存入y变量，作为目标向量

print(iris_X[:2,:])
print(iris_y)


# %%
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# 将iris_X和iris_y都分别按30%测试集的比例划分train集和test集

# 定义用到的模块
knn = KNeighborsClassifier()    # 使用knn模块训练数据分类
# knn = KNeighborsClassifier(n_neighbors=5)     # K近邻会将邻近点求平均，这里可指定平均邻近几个点的值
knn.fit(X_train, y_train)       # 使用的是fit函数

# 测试预测结果
print(knn.predict(X_test))
print(y_test)

# 可视化（自己增加）
yuc = knn.predict(X_test)   # 预测结果
zhs = y_test                # 实际值
import numpy as np 
idx = np.arange(0,len(yuc),1)     # 按元素数（len取值）生成索引，用于x坐标
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(idx,yuc,s=80,c='g',alpha=0.5)   # idx为x，预测和实际值为y
plt.scatter(idx,zhs,s=80,c='r',alpha=0.5)   # 设置图形足够大，颜色区分，有透明度


# %%
# 案例2. 本例波士顿房价数据集，使用linear_model实现线性回归预测

from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()    # 加载波士顿房价数据集
data_X = loaded_data.data               # 数据的data属性就是特征向量集
data_y = loaded_data.target             # 数据的target属性就是目标函数

model = LinearRegression()      # 使用线性回归模型
model.fit(data_X, data_y)

print(model.predict(data_X))
print(data_y)

# 可视化（自己增加）
yuc = model.predict(data_X)     # 预测结果
zhs = data_y                    # 实际值
import numpy as np 
idx = np.arange(0,len(yuc),1)       # 按元素数（len取值）成索引，用于x坐标
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.plot(idx,yuc,c='g',alpha=0.5)   # idx为x，预测和实际值为y
plt.plot(idx,zhs,c='r',alpha=0.5)   # 设置颜色区分，有透明度


# %%
# model模块的常见属性和功能，如上述的predict预测功能（1分类2回归）

model = LinearRegression()  # 指定本例所用的model
model.fit(X,y)      # 对特征向量集和目标向量，用模型进行拟合
model.predict(X)    # 对测试集数据X,用模型进行预测

model.coef_         # 模型的斜率
model.intercept_    # 模型的截距
model.get_params()  # 获得模型选择时给模型定义的参数
model.score(X,y)    # 对预测结果打分。用X预测，用y做真值进行比较。R^2方式打分


# %%
# 预处理preprocessing
# 标准化normalization、正则化、特征缩放feature scaling
# Idea: Make sure features are on a similar scale. 各特征处于相近的量级，便于学习

from sklearn import preprocessing
X = preprocessing.scale(X)  # 对数据进行预处理（标准化，缩放到0-1之间的数值）


# %%
# 交叉验证（数据集分割）

# 上面案例1中的数据集分割方式，按照固定比例分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 为了有效评价模型，对数据集进行多次不同模式的分割，分别测试并平均其准确率
from sklearn.model_selection import cross_val_score     # cross_val_score函数也并入model_selection
knn = KNeighborsClassifier(n_neighbors=5)               # 计算5个近邻点
score = cross_val_score(knn, X, y, cv=5, scoring='accuracy')    # 分类问题用准确率
# 打分由多次分割评估结果平均而来，使用knn模型，对X预测，用y验证，使用5种分割方案，打分使用准确率进行
loss = -cross_val_score(knn, X, y, cv=5, scoring='neg_mean_squared_error')  # 回归问题用均方差（原值时负值）
# 原mean_squared_error参数已弃用


# %%
# 学习率曲线，可视化学习的准确率变化过程
from sklearn.model_selection import learning_curve      # 学习曲线也放入model_selection
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np 

digits = load_digits()      # 加载数据集
X = digits.data             # digits属性作为特征向量集
y = digits.target           # 目标向量

# 学习曲线计算（指定阶段的准确率/损失值变化），输出给训练集大小、训练集损失、测试集损失等变量
# gamma是学习率（速率），阶段有数组指定，损失计算和上述交叉验证方法一样
train_sizes, train_loss, test_loss = learning_curve(
        SVC(gamma=0.001),X,y,cv=10,scoring='neg_mean_squared_error',
        train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
train_loss_mean = -np.mean(train_loss,axis=1)   # 上述cv10次分割的值求均值
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-',color='r', label="training")
plt.plot(train_sizes, test_loss_mean, 'o-',color='g', label="cross-validation")
plt.xlabel('training examples')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


# %%
# 模型调参过程，使用validation_curve评估参数取值变化过程中评估指标的变化曲线，根据是否欠拟合或过拟合来选取该参数的合适范围

from sklearn.model_selection import validation_curve    # 评估曲线也放入model_selection
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np 

digits = load_digits()
X = digits.data
y = digits.target

param_range = np.logspace(-6, -2.3, 10)          # 在区间取5个点，用于测试参数（调参）
# 评估曲线计算（指定阶段的准确率/损失值变化），输出给训练集大小、训练集损失、测试集损失等变量
# gamma是学习率（速率），阶段有数组指定，损失计算和上述交叉验证方法一样
train_loss, test_loss = validation_curve(       # 改用评估曲线，返回值没有train_sizes
                                                # SVC的固定参数去掉，后面给出参数名和取值范围（已指定）
        SVC(),X,y,param_name='gamma',param_range=param_range, cv=10,scoring='neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss,axis=1)   # 上述cv10次分割的值求均值
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(param_range, train_loss_mean, 'o-',color='r', label="training")
plt.plot(param_range, test_loss_mean, 'o-',color='g', label="cross-validation")
plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


# %%
# 保存model和参数
# pickle方法
import pickle
with open('/path/to/file.pickle','wb') as f:    # 打开句柄-写入
    pickle.dump(model,f)        # 保存模型
with open('/path/to/file.pickle','rb') as f:    # 打开句柄-读出
    mdl = pickle.load(f)        # 加载模型
    print(mdl.predict(X[0:1]))  # 使用模型预测

# joblib方法-sklearn
from sklearn.externals import joblib 
joblib.dump(model,'/path/to/file.pkl')  # 保存模型
mdl = joblib.load('/path/to/file.pkl')  # 加载模型
print(mdl.predict(X[0:1]))              # 使用模型预测


