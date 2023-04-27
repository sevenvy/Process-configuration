import numpy as np
import random
import itertools
import copy


'''第二代非支配遗传算法NSGA-Ⅱ'''
# NSGA-Ⅱ可参考：https://blog.csdn.net/weixin_44034444/article/details/119960596

# 模型参数设置
# N = 160 # 种群个体数量，需要为偶数以便交叉操作
# pm = 0.05  # 变异概率
# object_num = 4 # 优化目标数量
# H = 5 # 创建参考点时每个目标上的分段数量

'''与NSGA-Ⅲ相比，仅是个体的选择方式不同，其他相同'''

'''拥挤度计算与个体选择'''
# 输入：
# S1 前几个前沿合并形成的集合，恰好使得涉及的个体数量>=N，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]];
# fronts 前几个前沿的集合，其中每个二级元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]];
# N 种群中个体数量;
# 输出：
# A 所选的所有个体的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号]]
def crowding_distance(S1, fronts, N):
    # 在每个目标方向上按大小排序，为便于之后找到对应位置，对各目标函数值加上一列编号
    objects = []
    for k in range(len(S1[0][-1])): # 遍历每个目标方向
        B = [[S1[i][-1][k]] + [i] for i in range(len(S1))] # 依次取出第k个目标值，并在每个目标值后边加上编号，便于之后对应起来
        B.sort(key=lambda x:x[0])
        objects.append(B)

    # S1中除最后一个前沿的元素外，已涉及的个体统计
    num_last = len(fronts[-1]) # 最后一个前沿的元素数量
    A = []
    for i in range(len(S1)-num_last):
        if S1[i][0:2] not in A:
            A.append(S1[i][0:2])
    
    # 需要从最后一个前沿中选择的个体的数量
    K = N - len(A)

    # 筛选出最后一个前沿的前几个前沿中还未出现的个体相关的元素
    for i in range(len(S1)): # 对S1中目标函数值集合最后加上编号，便于查找
        S1[i][-1] += [i]
    S2 = list(filter(lambda x: x[0:2] not in A, S1[len(S1) - num_last:]))

    # 比较最后一个前沿中各元素的拥挤度，从中选取K个个体
    I = []
    for i in range(len(S2)):
        I_ks = 0
        for k in range(len(S2[0][-1])-1):
            if S2[i][-1][k] == objects[k][0][0] or S2[i][-1][k] == objects[k][-1][0]: # 如果是最大或最小值，则拥挤距离为无穷大，这里令其为目标总数，方便数学处理
                I_k = len(S2[0][-1])-1
            else:
                j = objects[k].index([S2[i][-1][k], S2[i][-1][-1]])
                I_k = (objects[k][j+1][0] - objects[k][j-1][0]) / (objects[k][-1][0] - objects[k][0][0])
            I_ks += I_k
        I.append(I_ks)
    for i in range(len(S2)):
        S2[i] += [I[i]] # 给每个元素后边加上其相应的拥挤度，便于绑定处理
    S2.sort(key=lambda x:x[-1])
    for _ in range(K):
        A.append(S2[0][0:2])
        S2 = list(filter(lambda x: x[0:2] != S2[0][0:2], S2)) # 过滤出其中还未包含在A中的元素，过滤器处理后元素顺序不变
    
    return A




