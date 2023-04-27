import numpy as np
import random
import itertools
import copy


'''多目标模拟退火算法'''
# 模拟退火算法可参考 https://blog.csdn.net/yanfeng1022/article/details/98519397
# 多目标模拟退火算法可参考 https://blog.csdn.net/qq_42364307/article/details/115128487

# 模型参数设置
# N_num = 50 # 生成的邻域解数量
# L = 20  # 每个温度水平扰动次数，也即迭代次数
# T0 = 100 # 初始温度
# Kt = 0.03 # 温度前的系数，称为Boltzmann常数
# MG1 = 400 # 总迭代次数
# MG2 = 250 # 最优解集持续未变化的最大迭代次数


'''邻域解生成操作'''
# 输入：
# N 生成的邻域解数量；parents 父代，每个解形式为[特征1对应五元组合，特征2对应五元组合，……]；i_j_P 父代种群中各加工特征所选取的五元组合编号的列表，每行形式为[特征1对应五元组合编号，特征2对应五元组合编号，……];
# Q_class 已按照特征进行过分类整理的五元组合，其每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F;
# 输出：
# childs 子代；i_j_C 子代种群中各加工特征所选取的五元组合编号的列表
def neighbour_generate(N_num, parents, i_j_P, Q_class):
    # 邻域解生成思路：随机选取一个解上的任意一个五元组合，随机替换成其他可行的五元组合，重复操作N_num次，如新生成的解与现有解重复则重新操作
    childs = []
    i_j_C = []
    for _ in range(N_num):
        k = 1 # 是否执行选择的标志
        while k == 1:
            a = random.sample(range(len(parents)), 1)[0]
            b = random.sample(range(len(parents[a])), 1)[0]
            R = list(filter(lambda x: x != parents[a][b], Q_class[b]))
            c = random.sample(R, 1)[0]
            i_j1 = i_j_P[a][0:b] + [c[0]] + i_j_P[a][b+1:]
            if i_j1 in i_j_P or i_j1 in i_j_C: # 如果随机选择的五元组合已存在，为避免重复，重新选择
                k = 1
            else:
                childs.append(parents[a][0:b] + [c[1:]] + parents[a][b+1:])
                i_j_C.append(i_j1)
                k = 0
    return childs, i_j_C


'''邻域解生成操作(新)'''
# 输入：
# solution_five 单个解，形式为[特征1对应五元组合，特征2对应五元组合，……]; solution_ij 单个解的五元组合对应编号，形式为[特征1对应五元组合编号，特征2对应五元组合编号，……];
# Q_class 已按照特征进行过分类整理的五元组合，其每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F;
# 输出：
# childs 子代；i_j_C 子代种群中各加工特征所选取的五元组合编号的列表
def neighbour_generate_new(solution_five, solution_ij, Q_class):
    # 邻域解生成思路：随机选取一个五元组合，随机替换成其他可行的五元组合
    a = random.sample(range(len(solution_ij)), 1)[0]
    R = Q_class[a].copy()
    R.remove([solution_ij[a]] + solution_five[a])
    b = random.sample(R, 1)[0]
    i_j_C = solution_ij[0:a] + [b[0]] + solution_ij[a+1:]
    child = (solution_five[0:a] + [b[1:]] + solution_five[a+1:])
    return child, i_j_C


'''非支配排序'''
# 输入：
# F_N 上一个前沿与邻域解集的集合，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]；
# objects_calculation 目标函数计算结果，其维度为[个体*设备数量配置方案*目标函数值]；
# SE_set 所有解的设备数量配置与布局方案(每个设备型号选择多少台设备，以及各台设备的位置)，其维度为[解*设备数量配置与布局方案*[每个设备型号选择的设备数量, 每个设备的位置]]；
# 输出：
# fronts[0] 最佳前沿，其中每个元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量与布局配置方案], [各目标函数值]]；
# As 各层前沿涉及的解； objects_A 最佳前沿中涉及的解对应的目标函数值；SE_A 最佳前沿中涉及的解对应的设备数量配置与布局方案
def ENS_4(F_N, i_j, objects_calculation, SE_set): # 采用高效非支配排序方法（ENS）
    # 将几个集合合并一下便于后续操作
    E = []
    for i in range(len(F_N)):
        for j in range(len(SE_set[i])):
            E.append([F_N[i], i_j[i], SE_set[i][j], objects_calculation[i][j]])

    # 根据第一个目标函数值对所有元素进行排序，如果第一个目标函数值相等，则根据第二个目标函数值排序，依此类推，如果全都相等，则任意排列
    def sort_key1(elem): # 设置一个函数，用于根据第一个目标函数值排序
        return elem[-1][0]
    
    def sort_key2(elem): # 设置一个函数，用于根据第二个目标函数值排序
        return elem[-1][1]

    def sort_key3(elem): # 设置一个函数，用于根据第三个目标函数值排序
        return elem[-1][2]

    def sort_key4(elem): # 设置一个函数，用于根据第四个目标函数值排序
        return elem[-1][3]
    
    E.sort(key=sort_key1) # 根据第一个目标函数值排序

    n1 = 0
    while n1 < len(E)-1: # 找出第一个目标函数值相等的连续几个元素
        if E[n1][-1][0] == E[n1+1][-1][0]: # 有两个相等
            A1 = [E[n1], E[n1+1]]
            k1 = 1
            while n1+k1 < len(E)-1 and E[n1+k1][-1][0] == E[n1+k1+1][-1][0]: # 判断是否还有三个或三个以上相等的
                A1.append(E[n1+k1+1])
                k1 += 1
            A1.sort(key=sort_key2) # 根据第二个目标函数值排序
            E[n1:n1+k1+1] = A1[:]

            n2 = 0
            while n2 < len(A1)-1: # 找出第二个目标函数值相等的连续几个元素
                if A1[n2][-1][1] == A1[n2+1][-1][1]: # 有两个相等
                    A2 = [A1[n2], A1[n2+1]]
                    k2 = 1
                    while n2+k2 < len(A1)-1 and A1[n2+k2][-1][1] == A1[n2+k2+1][-1][1]: # 判断是否还有三个或三个以上相等的
                        A2.append(A1[n2+k2+1])
                        k2 += 1
                    A2.sort(key=sort_key3) # 根据第三个目标函数值排序
                    E[n1+n2:n1+n2+k2+1] = A2[:]

                    n3 = 0
                    while n3 < len(A2)-1: # 找出第三个目标函数值相等的连续几个元素
                        if A2[n3][-1][2] == A2[n3+1][-1][2]: # 有两个相等
                            A3 = [A2[n3], A2[n3+1]]
                            k3 = 1
                            while n3+k3 < len(A2)-1 and A2[n3+k3][-1][2] == A1[n3+k3+1][-1][2]: # 判断是否还有三个或三个以上相等的
                                A3.append(A2[n3+k3+1])
                                k3 += 1
                            A3.sort(key=sort_key4) # 根据第四个目标函数值排序
                            E[n1+n2+n3:n1+n2+n3+k3+1] = A3[:] # 如果还有第四个目标函数值相等，则任意排列即可，这里就不再操作
                            n3 += k3 + 1
                        else:
                            n3 += 1
                    n2 += k2 + 1
                else:
                    n2 += 1   
            n1 += k1 + 1         
        else:
            n1 += 1

    # 将排好序的元素依次分配到各前沿中
    fronts = [[E[0]]] # 第一个元素必定在第一个前沿中
    for i in range(1,len(E)):
        k1 = 0 # 是否已经完成当前元素的前沿分配的标志
        for f in range(len(fronts)):
            k2 = 0 # 当前前沿是否存在支配解的标志
            for j in range(len(fronts[f])-1, -1, -1): # 支配关系确认，按照算法，从当前前沿中最后一个元素往前判断
                if E[i][-1][0] >= fronts[f][j][-1][0] and E[i][-1][1] >= fronts[f][j][-1][1] and E[i][-1][2] >= fronts[f][j][-1][2] and E[i][-1][3] >= fronts[f][j][-1][3] and E[i][-1] != fronts[f][j][-1]:
                    k2 = 1
                    break # 只要存在一个支配解，就退出循环
            if k2 == 0:
                fronts[f].append(E[i])
                k1 = 1
                break
        if k1 == 0: # 如果前边的前沿均存在支配元素，则将当前元素放到新的一个前沿中
            fronts.append([E[i]])    
    
    # 各层前沿涉及的解统计  
    As = [[fronts[0][0][0]]]
    i_j_A = [fronts[0][0][1]]
    for j in range(1, len(fronts[0])):
        if fronts[0][j][0] not in As[0]:
            As[0].append(fronts[0][j][0])
            i_j_A.append(fronts[0][j][1])
    
    for i in range(1, len(fronts)):
        As.append([fronts[i][0][0]])
        for j in range(1, len(fronts[i])):
            if fronts[i][j][0] not in As[i]:
                As[i].append(fronts[i][j][0])

    # 统计最佳前沿中涉及的解对应的目标函数值、设备数量配置与布局方案
    objects_A = [[] for _ in range(len(As[0]))] # 不能直接用 [[]] * len((As[0]))，否则创建出来的地址相同
    SE_A = [[] for _ in range(len(As[0]))]
    for a in fronts[0]:
            b = As[0].index(a[0])
            objects_A[b].append(a[-1])
            SE_A[b].append(a[2])
    
    return fronts[0], As, i_j_A, objects_A, SE_A # 输出的仅是最佳前沿，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]


'''非支配排序(新)'''
# 输入：
# F_N 上一个前沿与邻域解集的集合，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]；
# objects_calculation 目标函数计算结果，其维度为[个体*设备数量配置方案*目标函数值]；
# SE_set 所有解的设备数量配置与布局方案(每个设备型号选择多少台设备，以及各台设备的位置)，其维度为[解*设备数量配置与布局方案*[每个设备型号选择的设备数量, 每个设备的位置]]；
# 输出：
# fronts[0] 最佳前沿，其中每个元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量与布局配置方案], [各目标函数值]]；
# As 各层前沿涉及的解； objects_A 最佳前沿中涉及的解对应的目标函数值；SE_A 最佳前沿中涉及的解对应的设备数量配置与布局方案
def ENS_8(F_N, i_j, objects_calculation, SE_set): # 采用高效非支配排序方法（ENS）
    # 将几个集合合并一下便于后续操作
    E = []
    for i in range(len(F_N)):
        for j in range(len(SE_set[i])):
            E.append([F_N[i], i_j[i], SE_set[i][j], objects_calculation[i][j]])

    # 根据第一个目标函数值对所有元素进行排序，如果第一个目标函数值相等，则根据第二个目标函数值排序，依此类推，如果全都相等，则任意排列
    def sort_key1(elem): # 设置一个函数，用于根据第一个目标函数值排序
        return elem[-1][0]
    
    def sort_key2(elem): # 设置一个函数，用于根据第二个目标函数值排序
        return elem[-1][1]

    def sort_key3(elem): # 设置一个函数，用于根据第三个目标函数值排序
        return elem[-1][2]

    def sort_key4(elem): # 设置一个函数，用于根据第四个目标函数值排序
        return elem[-1][3]
    
    E.sort(key=sort_key1) # 根据第一个目标函数值排序

    n1 = 0
    while n1 < len(E)-1: # 找出第一个目标函数值相等的连续几个元素
        if E[n1][-1][0] == E[n1+1][-1][0]: # 有两个相等
            A1 = [E[n1], E[n1+1]]
            k1 = 1
            while n1+k1 < len(E)-1 and E[n1+k1][-1][0] == E[n1+k1+1][-1][0]: # 判断是否还有三个或三个以上相等的
                A1.append(E[n1+k1+1])
                k1 += 1
            A1.sort(key=sort_key2) # 根据第二个目标函数值排序
            E[n1:n1+k1+1] = A1[:]

            n2 = 0
            while n2 < len(A1)-1: # 找出第二个目标函数值相等的连续几个元素
                if A1[n2][-1][1] == A1[n2+1][-1][1]: # 有两个相等
                    A2 = [A1[n2], A1[n2+1]]
                    k2 = 1
                    while n2+k2 < len(A1)-1 and A1[n2+k2][-1][1] == A1[n2+k2+1][-1][1]: # 判断是否还有三个或三个以上相等的
                        A2.append(A1[n2+k2+1])
                        k2 += 1
                    A2.sort(key=sort_key3) # 根据第三个目标函数值排序
                    E[n1+n2:n1+n2+k2+1] = A2[:]

                    n3 = 0
                    while n3 < len(A2)-1: # 找出第三个目标函数值相等的连续几个元素
                        if A2[n3][-1][2] == A2[n3+1][-1][2]: # 有两个相等
                            A3 = [A2[n3], A2[n3+1]]
                            k3 = 1
                            while n3+k3 < len(A2)-1 and A2[n3+k3][-1][2] == A1[n3+k3+1][-1][2]: # 判断是否还有三个或三个以上相等的
                                A3.append(A2[n3+k3+1])
                                k3 += 1
                            A3.sort(key=sort_key4) # 根据第四个目标函数值排序
                            E[n1+n2+n3:n1+n2+n3+k3+1] = A3[:] # 如果还有第四个目标函数值相等，则任意排列即可，这里就不再操作
                            n3 += k3 + 1
                        else:
                            n3 += 1
                    n2 += k2 + 1
                else:
                    n2 += 1   
            n1 += k1 + 1         
        else:
            n1 += 1

    # 将排好序的元素依次分配到各前沿中
    fronts = [[E[0]]] # 第一个元素必定在第一个前沿中
    for i in range(1,len(E)):
        k1 = 0 # 是否已经完成当前元素的前沿分配的标志
        for f in range(len(fronts)):
            k2 = 0 # 当前前沿是否存在支配解的标志
            for j in range(len(fronts[f])-1, -1, -1): # 支配关系确认，按照算法，从当前前沿中最后一个元素往前判断
                if E[i][-1][0] >= fronts[f][j][-1][0] and E[i][-1][1] >= fronts[f][j][-1][1] and E[i][-1][2] >= fronts[f][j][-1][2] and E[i][-1][3] >= fronts[f][j][-1][3] and E[i][-1] != fronts[f][j][-1]:
                    k2 = 1
                    break # 只要存在一个支配解，就退出循环
            if k2 == 0:
                fronts[f].append(E[i])
                k1 = 1
                break
        if k1 == 0: # 如果前边的前沿均存在支配元素，则将当前元素放到新的一个前沿中
            fronts.append([E[i]])    
    
    return fronts[0] # 输出的仅是最佳前沿，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]























