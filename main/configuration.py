import pandas as pd
import time
import os
import random
import numpy as np
from NSGAIII_algorithm import NSGAIII
from AMOSA_algorithm import AMOSA
from MOPSO_algorithm import MOPSO
from MOAPSO_algorithm import MOAPSO


# 生成n个初始位置矢量以及速度矢量
def location_generate(j, n, v_initial):
    locations = []
    v = []
    for _ in range(n):
        k = 1  # 是否执行选择的标志
        while k == 1:
            loca = []
            v_1 = []
            for feature in range(len(j)):  # 对于每个加工特征分别随机选取五元组合
                loca.append(random.random())
                v_1.append(v_initial)
            if loca in locations:  # 如果随机选择的五元组合已存在，为避免重复，重新选择
                k = 1
            else:
                k = 0
        locations.append(loca)
        v.append(v_1)
    return locations, v


# 将位置转化为工艺参数集
def location_to_process(quinary, location):
    process = []
    for ort in range(len(location)):
        quinary_ort = quinary[quinary['J Feature'] == ort+1]
        quinary_ort = np.array(quinary_ort[['quinary combination index', 'J Feature', 'P Process', 'M Machine', 'T Tool']])
        size = len(quinary_ort)
        i = 1
        while location[ort] > i / size:
            i += 1
        process.append(quinary_ort[i-1].tolist())
    return process

'''数据读取'''
t1 = time.time()
print('Data loading ......')
# 特征集
j = pd.read_csv('../database/feature.csv')
# 工艺集
p = pd.read_csv('../database/process.csv')
# 设备集
m = pd.read_csv('../database/machine.csv')
# 刀具集
t = pd.read_csv('../database/tool.csv')
# 五元组合集
quinary = pd.read_csv('../database/quinary_combination.csv')
t2 = time.time()
print('Data loading is finished in ' + str(t2 - t1) + 's.')

max_run = 4  # 最大运行次数
N = 100  # 种群规模
v_initial = 0  # 初始速度
for run in range(1, max_run+1):
    print(1)
    print('第' + str(run) + '次运行', '......')
    locations, v = location_generate(j, N, v_initial)
    print(locations)
    initials = []
    for i in range(N):
        initials.append(location_to_process(quinary, locations[i]))
    print(locations)
    print(initials)
    NSGAIII(initials)
    AMOSA(initials)
    MOPSO(locations, v)
    MOAPSO(locations, v)

