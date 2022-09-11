import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import integrate

# 数据读入，整理为一维结构
resp = pd.read_csv("./resp_data_pred.csv")
resp = resp.iloc[:, 1]

# 寻峰 M1
# peaks, _ = find_peaks(x, height=0)
# 寻峰 M2，设置峰间距获得主要峰--[最实用]--
# peaks, _ = find_peaks(resp, distance=200)
# 寻峰 M3，设置峰凸起度筛选足够凸起的峰
# prominence 指峰值与该峰基线的垂直距离，两个参数限定 prominence 范围
# peaks, properties = find_peaks(x, prominence=(0.4, 1))
# 融合方法二三，筛选峰值
peaks, _ = find_peaks(resp, distance=200, prominence=(0.4, 1))

# 输出峰值索引号列表
print("peaks list below:\n", peaks)                 # 输出峰值索引号
print("how many peaks is here:\n", len(peaks))      # 计算峰值个数，4 个峰值，有 3 个区间
# print("the first and second elements are:\n", peaks[0], peaks[1])

# 取相邻两个点，计算峰值连线与曲线之间的面积
id_a = peaks[0]                 # 第一个峰值
id_b = peaks[1]                 # 第二个峰值
id_e = peaks[-1]                # 倒数第一个峰值
# print(id_a, id_b, id_e)

# print(x[ind_a], x[ind_b])
# id_sect = [id_a, id_b]          # 索引区间范围是坐标
# val_sect = [resp[id_a], resp[id_b]]   # 值区间范围是x按索引取值
# id_delt = id_b - id_a           # 索引差 -> 横边
# val_delt = max(val_sect) - min(val_sect)    # 值差 -> 纵边

# 定义峰值连线、resp曲线两个函数，以及绘制区间
# x = [id_a, id_b]
# ya = [resp[id_a], resp[id_b]]
ya = resp[peaks]
yb = resp

# 绘图
plt.plot(resp)             # 绘制原曲线
plt.plot(ya, color='r')    # 绘制峰值点
plt.plot(yb, color='r')    # 绘制峰值点
plt.show()

