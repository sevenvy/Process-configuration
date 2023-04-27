import copy
import random


# 参数设置
# max_num = 25 # 每个邻域解集生成方法最多生成的解的个数


'''邻域解集生成方法'''
'''方法1'''
# 随机五元组合变换P，其余保持不变，最多产生max_num个解
# 输入：
# solution 需要求其邻域的解，其形式为[[各加工特征的五元组合], [各五元组合编号]]
# Q_class 已按照特征进行过分类整理的五元组合，在每个特征维度下，每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
# 输出：
# solution_neighbours 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
def neighbour_P(solution, Q_class):
    solution_neighbours = []
    for i in range(len(solution[1])):
        # A = list(filter(lambda r: r[2] != solution[0][i][1] and r[3] == solution[0][i][2] and r[4] == solution[0][i][3] and r[5] == solution[0][i][4], Q_class[i])) # 筛选出除P外都一致的五元组合
        # B = copy.deepcopy(solution)
        # for j in A:
        #     B[0][i] = j[1:]
        #     B[1][i] = j[0]
        #     solution_neighbours.append(B)
        for r in Q_class[i]:
            if r[2] != solution[0][i][1] and r[3] == solution[0][i][2] and r[4] == solution[0][i][3] and r[5] == solution[0][i][4]: # 筛选出除P外都一致的五元组合
                B = copy.deepcopy(solution)
                B[0][i] = r[1:]
                B[1][i] = r[0]
                solution_neighbours += [B]
    return solution_neighbours

'''方法2'''
# 随机五元组合变换M，其余保持不变
# 输入：
# solution 需要求其邻域的解，其形式为[[各加工特征的五元组合], [各五元组合编号]]
# Q_class 已按照特征进行过分类整理的五元组合，在每个特征维度下，每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
# 输出：
# solution_neighbours 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
def neighbour_M(solution, Q_class):
    solution_neighbours = []
    for i in range(len(solution[1])):
    #     A = list(filter(lambda r: r[2] == solution[0][i][1] and r[3] != solution[0][i][2] and r[4] == solution[0][i][3] and r[5] == solution[0][i][4], Q_class[i])) # 筛选出除M外都一致的五元组合
    #     B = copy.deepcopy(solution)
    #     for j in A:
    #         B[0][i] = j[1:]
    #         B[1][i] = j[0]
    #         solution_neighbours.append(B)
        for r in Q_class[i]:
            if r[2] == solution[0][i][1] and r[3] != solution[0][i][2] and r[4] == solution[0][i][3] and r[5] == solution[0][i][4]: # 筛选出除P外都一致的五元组合
                B = copy.deepcopy(solution)
                B[0][i] = r[1:]
                B[1][i] = r[0]
                solution_neighbours += [B]
    return solution_neighbours

'''方法3'''
# 随机五元组合变换M，其余保持不变，最多产生max_num个解
# 输入：
# solution 需要求其邻域的解，其形式为[[各加工特征的五元组合], [各五元组合编号]]
# Q_class 已按照特征进行过分类整理的五元组合，在每个特征维度下，每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
# 输出：
# solution_neighbours 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
def neighbour_T(solution, Q_class):
    solution_neighbours = []
    for i in range(len(solution[1])):
        # A = list(filter(lambda r: r[2] == solution[0][i][1] and r[3] == solution[0][i][2] and r[4] != solution[0][i][3] and r[5] == solution[0][i][4], Q_class[i])) # 筛选出除T外都一致的五元组合
        # B = copy.deepcopy(solution)
        # for j in A:
        #     B[0][i] = j[1:]
        #     B[1][i] = j[0]
        #     solution_neighbours.append(B)
        for r in Q_class[i]:
            if r[2] == solution[0][i][1] and r[3] == solution[0][i][2] and r[4] != solution[0][i][3] and r[5] == solution[0][i][4]: # 筛选出除P外都一致的五元组合
                B = copy.deepcopy(solution)
                B[0][i] = r[1:]
                B[1][i] = r[0]
                solution_neighbours += [B]
    return solution_neighbours

'''方法4'''
# 随机五元组合变换M，其余保持不变，最多产生max_num个解
# 输入：
# solution 需要求其邻域的解，其形式为[[各加工特征的五元组合], [各五元组合编号]]
# Q_class 已按照特征进行过分类整理的五元组合，在每个特征维度下，每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
# 输出：
# solution_neighbours 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
def neighbour_F(solution, Q_class):
    solution_neighbours = []
    for i in range(len(solution[1])):
        # A = list(filter(lambda r: r[2] == solution[0][i][1] and r[3] == solution[0][i][2] and r[4] == solution[0][i][3] and r[5] != solution[0][i][4], Q_class[i])) # 筛选出除T外都一致的五元组合
        # B = copy.deepcopy(solution)
        # for j in A:
        #     B[0][i] = j[1:]
        #     B[1][i] = j[0]
        #     solution_neighbours.append(B)
        for r in Q_class[i]:
            if r[2] == solution[0][i][1] and r[3] == solution[0][i][2] and r[4] == solution[0][i][3] and r[5] != solution[0][i][4]: # 筛选出除P外都一致的五元组合
                B = copy.deepcopy(solution)
                B[0][i] = r[1:]
                B[1][i] = r[0]
                solution_neighbours += [B]
    return solution_neighbours


'''新方法'''
# 输入：
# solution 需要求其邻域的解，其形式为[[各加工特征的五元组合], [各五元组合编号]]
# Q_class 已按照特征进行过分类整理的五元组合，在每个特征维度下，每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
# 输出：
# solution_neighbours 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
def neighbour_new(solution, Q_class):
    solution_neighbours = []
    for i in range(len(solution[0])):
        for j in range(1, len(solution[0][0])):
            for r in Q_class[i]:
                if r[1:j+1] == solution[0][i][0:j] and r[j+2:6] == solution[0][i][j+1:5] and r[j+1] != solution[0][i][j]: # 筛选出除第j个元素外都一致的五元组合
                    B = copy.deepcopy(solution)
                    B[0][i] = r[1:]
                    B[1][i] = r[0]
                    solution_neighbours += [B]
    return solution_neighbours


# def neighbour_new(solution, Q_class):
#     # solution_neighbours = []

#     a = random.sample(range(len(parents)), 1)[0]
#     b = random.sample(range(len(parents[a])), 1)[0]

#     for i in range(len(solution[0])):
#         for j in range(1, len(solution[0][0])):
#             for r in Q_class[i]:
#                 if r[1:j+1] == solution[0][i][0:j] and r[j+2:6] == solution[0][i][j+1:5] and r[j+1] != solution[0][i][j]: # 筛选出除第j个元素外都一致的五元组合
#                     B = copy.deepcopy(solution)
#                     B[0][i] = r[1:]
#                     B[1][i] = r[0]
#                     solution_neighbours += [B]
#     return solution_neighbours



'''适应度计算在 configuration_algorithm.py 中组织，具体函数见 optimization_objects.py'''


'''非支配排序'''
# 输入：
# F_N 上一个前沿与邻域解集的集合，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]；
# objects_calculation 目标函数计算结果，其维度为[个体*设备数量配置方案*目标函数值]；
# SE_set 所有解的设备数量配置与布局方案(每个设备型号选择多少台设备，以及各台设备的位置)，其维度为[解*设备数量配置与布局方案*[每个设备型号选择的设备数量, 每个设备的位置]]；
# 输出：
# front 最佳前沿，其中每个元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量与布局配置方案], [各目标函数值]]
# A 最佳前沿中涉及的解的信息，每行包含[[各加工特征的五元组合], [各五元组合编号]]
def ENS_1(F_N, objects_calculation, SE_set): # 采用高效非支配排序方法（ENS）
    # 将几个集合合并一下便于后续操作
    E = []
    for i in range(len(F_N)):
        for j in range(len(SE_set[i])):
            E.append([F_N[i][0], F_N[i][1], SE_set[i][j], objects_calculation[i][j]])

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

    # 在排好序的元素中获取最佳前沿
    front = [E[0]]
    for i in range(1,len(E)):
        k = 0
        for j in range(len(front)-1, -1, -1): # 支配关系确认，按照算法，从当前前沿中最后一个元素往前判断
            if E[i][-1][0] >= front[j][-1][0] and E[i][-1][1] >= front[j][-1][1] and E[i][-1][2] >= front[j][-1][2] and E[i][-1][3] >= front[j][-1][3] and E[i][-1] != front[j][-1]:
                k = 1
                break # 只要存在一个支配解，就退出循环
        if k == 0:
            front.append(E[i])
        
    # 统计最佳前沿中涉及的解以及整理对应的目标函数值、设备数量配置与布局方案
    A = []
    objects_A = []
    SE_A = []
    for a in front:
        if a[0:2] not in A:
            A.append(a[0:2])
            objects_A.append([a[-1]])
            SE_A.append([a[2]])
        else:
            b = A.index(a[0:2])
            objects_A[b].append(a[-1])
            SE_A[b].append(a[2])

    return front, A, objects_A, SE_A # 输出的仅是最佳前沿，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]


'''非支配排序(仅以前沿作为输入)'''
# 输入：
# front_set 不同前沿的集合，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]；
# 输出：
# front 最佳前沿，其中每个元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量与布局配置方案], [各目标函数值]]
def ENS_6(front_set): # 采用高效非支配排序方法（ENS）
    E = front_set.copy()
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

    # 在排好序的元素中获取最佳前沿
    front = [E[0]]
    for i in range(1,len(E)):
        k = 0
        for j in range(len(front)-1, -1, -1): # 支配关系确认，按照算法，从当前前沿中最后一个元素往前判断
            if E[i][-1][0] >= front[j][-1][0] and E[i][-1][1] >= front[j][-1][1] and E[i][-1][2] >= front[j][-1][2] and E[i][-1][3] >= front[j][-1][3] and E[i][-1] != front[j][-1]:
                k = 1
                break # 只要存在一个支配解，就退出循环
        if k == 0:
            front.append(E[i])
        
    return front # 输出的仅是最佳前沿，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]