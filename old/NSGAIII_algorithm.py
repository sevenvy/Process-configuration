import numpy as np
import random
import itertools
import copy


'''第三代非支配遗传算法NSGA-Ⅲ'''
# NSGA-Ⅲ可参考：https://blog.csdn.net/ztzi321/article/details/111304393

# 模型参数设置
# N = 160 # 种群个体数量，需要为偶数以便交叉操作
# pm = 0.05  # 变异概率
# object_num = 4 # 优化目标数量
# H = 5 # 创建参考点时每个目标上的分段数量


'''随机生成初始种群'''
# 输入：
# Q_class 已按照特征进行过分类整理的五元组合，其每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
# N 种群个体数量
# 输出：
# initials 初始种群，每个个体形式为[特征1对应五元组合，特征2对应五元组合，……]；i_j 初始种群中各加工特征所选取的五元组合编号的列表，每个个体形式为[特征1对应五元组合编号，特征2对应五元组合编号，……]
def generate_initials(Q_class, N):
    initials = []
    i_j = []
    for n in range(N):
        k = 1 # 是否执行选择的标志
        while k == 1:
            initial1 = []
            i1_j = []
            for j in range(len(Q_class)): # 对于每个加工特征分别随机选取五元组合
                A = random.sample(Q_class[j],1)[0]
                initial1.append(A[1:])
                A_index = Q_class[j].index(A)
                i1_j.append(Q_class[j][A_index][0])
            if initial1 in initials: # 如果随机选择的五元组合已存在，为避免重复，重新选择
                k = 1
            else:
                k = 0
        initials.append(initial1)
        i_j.append(i1_j)
    return initials, i_j


'''交叉与变异操作'''
# 输入：
# N 种群个体数量；parents 父代；pm 变异概率；i_j 初始种群中各加工特征所选取的五元组合编号的列表;
# Q_class 已按照特征进行过分类整理的五元组合，其每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F;
# 输出：
# childs 子代；i_j1 子代种群中各加工特征所选取的五元组合编号的列表
def crossover_mutation(N, parents, pm, Q_class, i_j):
    childs = []
    i_j1 = []
    # 交叉操作
    # 对父代个体进行随机两两分组，每组进行一次交叉形成两个新的子代个体
    # print(parents[0])
    # print(i_j[0])
    P_I = list(zip(parents, i_j))
    # print(P_I[0])
    random.shuffle(P_I) # 打乱父代个体及其编号的顺序，之后按顺序两两交叉，也相当于随机分组后交叉
    parents, i_j = zip(*P_I)
    parents = list(parents)
    # print(parents[0])
    i_j = list(i_j)
    # print(i_j[0])
    for n in range(int(N/2)):  # 当子代种群个体数量达到设定总数时停止迭代，由于每次交叉会产生2个子代个体，因此为N/2
        cross_point = random.randint(1, len(parents[0]) - 1) # 随机给定交叉点，此处父代中第cross_point个五元组合及其之前的五元组合保留，此后的五元组合则相互交换
        child1 = parents[2*n][:cross_point] + parents[2*n+1][cross_point:]
        child2 = parents[2*n+1][:cross_point] + parents[2*n][cross_point:]
        i_j_1 = i_j[2*n][:cross_point] + i_j[2*n+1][cross_point:]
        i_j_2 = i_j[2*n+1][:cross_point] + i_j[2*n][cross_point:]
        childs.append(child1)
        childs.append(child2)
        i_j1.append(i_j_1)
        i_j1.append(i_j_2)

    # 变异操作
    for n in range(N):
        for j in range(len(childs[n])):
            if random.random() < pm:  # 以一定概率进行变异操作
                A = [i_j1[n][j]] + childs[n][j] # 在子代个体的相应五元组合前加上相应的五元组合编号
                # print(A)
                B = copy.deepcopy(Q_class[j]) # 复制一下，避免直接赋值情况下两者共用索引，从而导致使用remove或del时，将两者共用的索引删去（remove或del是删除元素的索引）
                # print(j)
                # if A not in B:
                # print(i_j1[n])
                # print(childs[n])
                # E = []
                # for k1 in range(len(childs[n])):
                #     for k2 in range(len(Q_class[k1])):
                #         if childs[n][k1] == Q_class[k1][k2][1:]:
                #             E.append(Q_class[k1][k2][0])
                # print(E)
                    # for k in range(len(B)):
                    #     if A[1:] == B[k][1:]:
                    #         print(B[k])
                B.remove(A) # 得到除原五元组合外的其他五元组合的集合
                C = random.sample(B,1)[0]
                childs[n][j] = C[1:]
                i_j1[n][j] = C[0]

    # 强制变异操作
    # 如果存在与父代或其他子代相同的个体，则强制其随机一个五元组合进行变异操作，并且保证无重复
    for n in range(N):
        D = copy.deepcopy(childs + parents) # 复制一下，避免直接赋值情况下两者共用索引，从而导致使用remove或del时，将两者共用的索引删去（remove或del是删除元素的索引）
        D.remove(D[n]) # 去掉当前个体
        Dn = childs[n]
        k1 = 1 # 是否需要判断是否存在相同项的标志
        k2 = 0 # 是否进行了强制变异的标志
        while k1 == 1:
            if Dn in D: # 如果仍然存在相同项
                # print(1)
                Dn = copy.deepcopy(childs[n])
                a = random.randint(0, len(childs[n]) - 1) # 随机选择个体中的一个五元组合进行变异
                A = [i_j1[n][a]] + childs[n][a] # 在子代个体的相应五元组合前加上相应的五元组合编号
                B = copy.deepcopy(Q_class[a])
                B.remove(A) # 得到除原五元组合外的其他五元组合的集合
                C = random.sample(B,1)[0]
                Dn[a] = C[1:]
                k2 = 1
            else:
                k1 = 0
                if k2 == 1: # 如果进行了强制变异，则将变异后的子代个体更新
                    childs[n] = copy.deepcopy(Dn)
                    i_j1[n][a] = C[0]

    return childs, i_j1


'''适应度计算在 configuration_algorithm.py 中组织，具体函数见 optimization_objects.py'''


'''非支配排序'''
# 输入：
# N 种群个体数量；P_C 父代与子代的集合；objects_calculation 目标函数计算结果，其维度为[个体*设备数量配置方案*目标函数值]；i_j 父代与子代的集合中各加工特征所选取的五元组合编号的列表;
# SE_set 所有个体的设备数量配置与布局方案(每个设备型号选择多少台设备，以及各台设备的位置)，其维度为[个体*设备数量配置与布局方案*[每个设备型号选择的设备数量, 每个设备的位置]]；
# 输出：
# fronts 前几个前沿的集合，其中每个二级元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量与布局配置方案], [各目标函数值]]
# num 已划分完成的前沿中所涉及的个体总数；A 目前分配好的前沿中已经涉及的个体信息，每行包含[[各加工特征的五元组合], [各五元组合编号]]
def ENS(N, P_C, SE_set, i_j, objects_calculation): # 采用高效非支配排序方法（ENS）
    # 将几个集合合并一下便于后续操作
    E = []
    for i in range(len(P_C)):
        for j in range(len(SE_set[i])):
            E.append([P_C[i], i_j[i], SE_set[i][j], objects_calculation[i][j]])

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

    # 只取前几个涉及的个体总数已经达到要求的种群个体数量的前沿
    A = [] # 用于储存前几个前沿前沿中涉及的个体信息
    num = 0 # 前几个前沿中所涉及的个体总数
    for a in range(len(fronts)):
        for b in fronts[a]:
            if b[0:2] not in A:
                num += 1
                A.append(b[0:2])
        if num >= N: # 如果前几个前沿中涉及的个体总数已经达到要求的种群个体数量，那么就停止搜寻，给出前几个前沿
            break

    return fronts[0:a+1], num, A # 输出的仅是前几个前沿的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]


'''目标函数解空间超平面的自适应标准化'''
# 输入：
# S1 前几个前沿合并形成的集合，恰好使得涉及的个体数量>=N，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]
# 输出：
# S1 经过自适应标准化的前几个前沿合并形成的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [标准化的各目标函数值]]
def nomalization(S1):
    # 理想点计算与目标函数值偏移
    for j in range(len(S1[0][-1])): # 遍历每个目标
        ideal_point = min(A[-1][j] for A in S1) # 求每个目标函数的理想点（可以理解成，以此作为标准化的参考原点）
        for i in range(len(S1)):
            S1[i][-1][j] -= ideal_point # 对每个目标函数值减去理想点（可以理解成，将标准点迁移到坐标系原点）
        
    # 获取每个目标的最大值，作为极端点计算的参考
    object_max = []
    for j in range(len(S1[0][-1])):
        a = max(A[-1][j] for A in S1)
        if a == 0:
            a = 1e-6 # 万一出现各个体的同一目标的值都一样, 从而出现减去ideal_point后全为0的情况, 就令其为一个较小的值, 从而避免后边作为除数为0
        object_max.append(a)
    
    # 极端点计算
    extreme_points = []
    for j in range(len(S1[0][-1])):
        axis = [l for l in range(len(S1[0][-1]))].copy()
        axis.remove(j) # 除当前维度的方向外，其他维度方向都要进行计算
        A = []
        for n in range(len(S1)):
            A.append(max(S1[n][-1][k] / object_max[k] for k in axis)) # 除当前维度的方向外，其他维度方向都计算并求最大值
        extreme_points.append(S1[A.index(min(A))]) # 每个维度方向上的极端点即为在其他维度方向上的值均很小的解（由于非支配关系的原因，其他维度方向上的值很小时，当前维度上的值必定相对较大）

    # 截距计算
    B = []
    for j in range(len(extreme_points)): # 把目标函数值剥离出来
        B.append(extreme_points[j][-1])
    B = np.array(B)
    
    k = 1
    while k == 1:
        try:
            B1 = np.linalg.inv(B) # 求逆
        except:
            for s in range(len(B)):
                B[s,s] += object_max[s] / 100 # 为避免出现矩阵不可逆的情况，万一出现不可逆，则将斜对角上每个元素加一个较小的值（可以理解成，将与坐标轴平行的超平面在三个方向上都偏一点点，从而使得存在截距）
        else:
            a = np.dot(B1, np.ones((len(extreme_points),1))) # B*a=C，则a=B的逆*C，这里C为（4，1）的所有元素为1的矩阵，整个解空间的超平面的方程即为 a * x1 + b * x2 + c * x3 + d * x4 = 1
            # 实际上a中四个元素的倒数才是各维度方向的截距，但标准化时是各目标值除以截距，这里就不再取倒数，后边则改成乘
            k = 0

    a = list(zip(*a))[0] # 通过上述计算，a为np数组，这里转换成元组（tuple）

    # 各目标的标准化
    for n in range(len(S1)):
        for j in range(len(a)):
            S1[n][-1][j] *= a[j]

    return S1


'''参考点创建'''
# 输入：
# object_num 优化目标数量；H：创建参考点时每个目标上的分段数量；
# 输出：
# referpoint_set 参考点集合
def refer_points(object_num, H): # 具体可参考介绍：https://blog.csdn.net/ztzi321/article/details/111304393
    # 构建M-1维的组合
    s = []
    for j in range(H+object_num-1):
        s.append(round(j / H, 2))
    X = []
    for combin in itertools.combinations(s, object_num-1):
        X.append(list(combin))
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = round(X[i][j] - j / H, 2) # 这里不是(j-1)/H的原因是，这里i和j编号都是从0开始的，而算法里是从1开始的
    # 创建参考点
    referpoint_set = []
    for i in range(len(X)):
        referpoint = []
        for j in range(object_num):
            if j == 0: # 这里也是编号从0开始的原因，不用j=1
                S = X[i][j]
            elif j > 0 and j < object_num - 1:
                S = round(X[i][j] - X[i][j-1], 2)
            elif j == object_num - 1:
                S = round(1 - X[i][j-1], 2)
            else:
                print('Error with the refer points creation.')
                exit(0)
            referpoint.append(S)
        referpoint_set.append(referpoint)

    return referpoint_set


'''各点与参考点距离计算'''
# 输入：
# S1 经过自适应标准化的前几个前沿合并形成的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [标准化的各目标函数值]]；referpoint_set 参考点集合；
# 输出：
# nearest_point 与每个元素距离最小的参考点；nearest_d 与每个元素的最小距离
def distance(S1, referpoint_set):
    # 计算每个元素与每个参考点间的余弦距离
    nearest_d = []
    nearest_point = []
    for i in range(len(S1)):
        d = []
        for j in range(len(referpoint_set)):
            A = np.linalg.norm(S1[i][-1]) * np.linalg.norm(referpoint_set[j])
            if A < 1e-6:
                A = 1e-6 # 万一出现值很小的情况, 就令其为一个较小的值, 从而避免后边作为除数为0
            d.append(1 - np.dot(S1[i][-1], referpoint_set[j]) / A) # 此处采用的是余弦距离，可参考：https://blog.csdn.net/lucky_kai/article/details/89514868
        
        # 获取与每个个体的最小距离及相应的参考点A
        nearest_d.append(min(d))
        nearest_point.append(referpoint_set[d.index(min(d))])

    return nearest_point, nearest_d
    

'''最后一个前沿个体筛选'''
# 输入：
# N 种群中个体数量；fronts 前几个前沿的集合，其中每个二级元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]];
# S1 经过自适应标准化的前几个前沿合并形成的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [标准化的各目标函数值]]；referpoint_set 参考点集合；
# nearest_point 与每个元素距离最小的参考点；nearest_d 与每个元素的最小距离;
# 输出：
# A 所选的所有个体的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号]]
def last_selection(N, S1, fronts, referpoint_set, nearest_point, nearest_d):
    # 统计除S1中包含的最后一个前沿外，前边的前沿中的元素最接近的参考点的出现次数
    num_last = len(fronts[-1]) # 最后一个前沿的元素数量
    num_nearest = [0 for _ in range(len(referpoint_set))]
    A = [] # 用于记录已涉及的个体的信息
    for i in range(len(S1)-num_last):
        num_nearest[referpoint_set.index(nearest_point[i])] += 1
        if S1[i][0:2] not in A:
            A.append(S1[i][0:2])

    # 将最后一个前沿的前几个前沿中已经出现的个体相关的元素全都去除
    a = 0
    for i in range(len(S1)-num_last, len(S1)):
        b = i - a
        if S1[b][0:2] in A:
            del(S1[b])
            a += 1
            num_last -= 1

    # 需要从最后一个前沿中选择的个体的数量
    K = N - len(A)

    # 最后一个前沿中涉及的个体的选择
    k = 1
    while k <= K:
        referpoint_small = []
        for i in range(len(referpoint_set)): # 获取出现次数最少的参考点的集合
            if num_nearest[i] == min(num_nearest):
                referpoint_small.append(referpoint_set[i])
        point = random.sample(referpoint_small,1)[0] # 随机从出现次数最少的参考点中抽取一个

        I = []
        I_d = []
        for i in range(len(S1)-num_last, len(S1)): # 找出最后一个前沿中出现次数最少的所选参考点最接近的元素
            if nearest_point[i] == point:
                I.append(S1[i])
                I_d.append(nearest_d[i])
        
        if I == []: # 如果不存在则将这个参考点去掉，从剩下的参考点中选择
            del(num_nearest[referpoint_set.index(point)])
            referpoint_set.remove(point)
        else:
            if min(num_nearest) == 0: # 若存在此参考点，且参考点的出现次数为0，则选择距离较小的种群个体
                B = I[I_d.index(min(I_d))]
            else: # 若出现次数不为0，由于距离与前边无法对比，任意选择一个
                B = random.sample(I,1)[0]
            A.append(B[0:2])
            num_nearest[referpoint_set.index(point)] += 1
            
            # 将最后一个前沿中涉及当前个体的元素全都去除
            a = 0
            for i in range(len(S1)-num_last, len(S1)):
                b = i - a
                if S1[b][0:2] == B[0:2]:
                    del(S1[b])
                    a += 1
                    num_last -= 1

            k += 1

    return A


'''更新父代'''
# 输入：
# S1 前几个前沿合并形成的集合，恰好使得涉及的个体数量>=N，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]];
# A 所选的所有个体的集合，其中包含 [[各加工特征的五元组合], [各五元组合编号]]
def update(S1, A):
    parents = [a[0] for a in A]
    i_j_P = [a[1] for a in A]
    objects_calculation_P = []
    SE_set_P = []
    for i in range(len(A)):
        B = list(filter(lambda r: r[0:2] == A[i], S1))
        objects_calculation_P.append([b[-1] for b in B])
        SE_set_P.append([b[2] for b in B])
    return parents, i_j_P, objects_calculation_P, SE_set_P


'''非支配排序（用于全局最优前沿更新）'''
# 输入：
# E 每行形式为[[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]
def ENS_best(E): # 采用高效非支配排序方法（ENS）
    # 根据第一个目标函数值对所有元素进行由小到大排序，如果第一个目标函数值相等，则根据第二个目标函数值排序，依此类推，如果全都相等，则任意排列
    object_num = len(E[0][-1]) # 优化目标数量
    for k in range(object_num):
        if k == 0:
            E.sort(key=lambda x:x[-1][0]) # 根据第一个目标函数值排序
        else:
            i = 0
            while i < len(E):
                head = E[i][-1][0:k]
                j = i
                while j < len(E) and E[j][-1][0:k] == head:
                    j += 1
                E1 = E[i:j]
                E1.sort(key=lambda x:x[-1][k])
                E[i:j] = E1
                i = j

    # 在排好序的元素中获取最优前沿
    front = [E[0]]
    for i in range(1,len(E)):
        for j in range(len(front)-1, -1, -1): # 支配关系确认，按照算法，从当前前沿中最后一个元素往前判断
            flag = 0
            if E[i][-1] != front[j][-1]:
                for k in range(1,object_num):
                    if E[i][-1][k] < front[j][-1][k]: # 但凡待分配的元素有一个目标值小于前沿中的元素的对应目标值，就比定不会被此前沿中元素支配
                        flag = 1
                        break
            if flag == 0:
                break # 只要存在一个支配解，就退出循环
        if flag == 1:
            front.append(E[i])

    # 统计最优前沿中涉及的个体
    A = []
    for a in front:
        if a[0:2] not in A:
            A.append(a[0:2])

    return front, A
