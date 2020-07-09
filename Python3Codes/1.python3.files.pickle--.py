#!/usr/bin/python3
# python3.files.pickle

# 导入pickle包
import pickle

# 定义输入数据和输出格式
data1 = {'a': [1, 2.0, 3, 4+6j],  # 定义数据对象，共3对键值对的字典
         'b': ('string', u'Unicode string'),
         'c': None}
selfref_list = [1, 2, 3]      # selfref_list给出自引用的数据的index
selfref_list.append(selfref_list) # 重复append自身数据一次，生成新的数据？？？
# pickle.dump
output = open('data.pkl','wb')
pickle.dump(data1, output)    # 将data数据传给output
pickle.dump(selfref_list, output, -1)
output.close()


# 用pickle模块重构python对象
# pickle.load
data1 = pickle.load(pkl_file) # 从pkl_file句柄加载数据
pprint.pprint(data1)          # pretty print格式化输出
pkl_file.close()              # 关闭pkl_file的文件句柄
