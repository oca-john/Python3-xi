#!/usr/bin/python3
# join()用于字串串接

# 连接符和字串列表预先定义
j = '.'                                         # 定义连接符为'.'
sep = ['one','two','three','four','five','six'] # 定义待连接的字串列表
out = j.join(sep)                               # 调用j连接符，join连接，sep字串列表中的字串集

# 连接符和字串列表即时定义
out = '.'.join(['one','two','three','four'])    # 连接操作时，即时定义连接符和字串集（可以预定义其中一项或两项）
