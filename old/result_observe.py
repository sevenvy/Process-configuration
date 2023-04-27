import csv
import os
from main.algrithm_compare import hypervolume, front_read, progress_bar, object_read, time_read


# 规范化处理
# 定义获取最大最小目标值函数
def min_max(object_front): # 以最佳前沿的各目标值集合作为输入
    min_objects = []
    max_objects = []
    for k in range(len(object_front[0])):
        min_objects.append(min([a[k] for a in object_front]))
        max_objects.append(max([a[k] for a in object_front]))
    return min_objects, max_objects

# 定义规范化处理函数
def object_normalize(object_front, min_overall, max_overall): # 以最佳前沿的各目标值集合、所有前沿的各目标的最小值和最大值集合作为输入
    object_front_nor = [[None for _ in range(len(object_front[0]))] for _ in range(len(object_front))]
    for k in range(len(object_front[0])):
        delta = max_overall[k] - min_overall[k]
        for i in range(len(object_front)):
            object_front_nor[i][k] = (object_front[i][k] - min_overall[k]) / delta
    return object_front_nor

# 定义大小反转函数（每个目标值都用1减一下，用于求超体积）
def object_reverse(object_nor):
    object_nor_re = [[None for _ in range(len(object_nor[0]))] for _ in range(len(object_nor))]
    for k in range(len(object_nor)):
        for l in range(len(object_nor[k])):
            object_nor_re[k][l] = 1 - object_nor[k][l]
    return object_nor_re

""""""
"""多次计算同时观察各项评估指标"""
""""""
test_num_total = 4 # 一共四次计算
# feature_solution_num_total = 4 # 一共四种特征划分方案

for i in range(1, test_num_total+1):
#     algo_results = [[] for _ in range(4)] # 用于记录每个算法的结果，一共有四种算法
#     for j in range(1, feature_solution_num_total+1):
          '''读取多目标函数值数据'''
#         # NSGAIII
#         NSGA3_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_NSGAIII_600_' + str(i) + '.csv')
#         NSGA3_data = front_read(NSGA3_path)
#         NSGA3_objects_X = [a[-1] for a in NSGA3_data]
#         # NSGAII
#         NSGA2_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_NSGAII_600_' + str(i) + '.csv')
#         NSGA2_data = front_read(NSGA2_path)
#         NSGA2_objects_X = [a[-1] for a in NSGA2_data]
#         # NS
#         NS_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_NS_600_' + str(i) + '.csv')
#         NS_data = front_read(NS_path)
#         NS_objects_X = [a[-1] for a in NS_data]
        # MOSA
        MOSA_path = os.path.join('results_all', 'results_test' + str(i), 'best_front_MOSA_600_' + str(i) + '.csv')
        MOSA_data = front_read(MOSA_path)
        MOSA_objects_X = [a[-1] for a in MOSA_data]

         '''读取各算法的计算时间'''
#         # NSGAIII
#         NSGA3_time_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'time_NSGAIII_600_' + str(i) + '.csv')
#         NSGA3_time = time_read(NSGA3_time_path)
#         # NSGAII
#         NSGA2_time_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'time_NSGAII_600_' + str(i) + '.csv')
#         NSGA2_time = time_read(NSGA2_time_path)
#         # NS
#         NS_time_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'time_NS_600_' + str(i) + '.csv')
#         NS_time = time_read(NS_time_path)
         # MOSA
         MOSA_time_path = os.path.join('results_all', 'results_test' + str(i), 'time_MOSA_600_' + str(i) + '.csv')
         MOSA_time = time_read(MOSA_time_path)
        
         '''超体积比较(越大越好)'''
#         # 求各优化目标方向上的全局极值
#         min_NSGA3, max_NSGA3 = min_max(NSGA3_objects_X)
#         min_NSGA2, max_NSGA2 = min_max(NSGA2_objects_X)
#         min_NS, max_NS = min_max(NS_objects_X)
         min_MOSA, max_MOSA = min_max(MOSA_objects_X)
         min_overall = [min_MOSA[k] for k in range(len(min_MOSA))]
         max_overall = [max_MOSA[k] for k in range(len(max_MOSA))]
         # 规范化
#         NSGA3_normalize = object_normalize(NSGA3_objects_X, min_overall, max_overall)
#         NSGA2_normalize = object_normalize(NSGA2_objects_X, min_overall, max_overall)
#         NS_normalize = object_normalize(NS_objects_X, min_overall, max_overall)
#         MOSA_normalize = object_normalize(MOSA_objects_X, min_overall, max_overall)
#         # 大小反转（原本各指标越小越好，但为便于求超体积，这里反转成越大越好）
#         NSGA3_reverse = object_reverse(NSGA3_normalize)
#         NSGA2_reverse = object_reverse(NSGA2_normalize)
#         NS_reverse = object_reverse(NS_normalize)
#         MOSA_reverse = object_reverse(MOSA_normalize)
#         # 求超体积
#         NSGA3_volume = hypervolume(NSGA3_reverse)
#         NSGA2_volume = hypervolume(NSGA2_reverse)
#         NS_volume = hypervolume(NS_reverse)
#         MOSA_volume = hypervolume(MOSA_reverse)

#         '''与全局最优前沿的平均距离比较（越小越好）'''
#         # 获取规范化后的全局最佳前沿
#         objects_nor_overall = NSGA3_normalize + NSGA2_normalize + NS_normalize + MOSA_normalize
#         front_overall = ENS_3(objects_nor_overall)
#         # 求平均距离
#         NSGA3_IGD = IGD(front_overall, NSGA3_normalize)
#         NSGA2_IGD = IGD(front_overall, NSGA2_normalize)
#         NS_IGD = IGD(front_overall, NS_normalize)
#         MOSA_IGD = IGD(front_overall, MOSA_normalize)

#         '''未被其他算法得到的前沿支配的元素的比例比较（越大越好）'''
#         # 统计未被其他算法得到的前沿支配的元素的数量
#         NSGA3_DPO_num = DPO(front_overall, NSGA3_normalize)
#         NSGA2_DPO_num = DPO(front_overall, NSGA2_normalize)
#         NS_DPO_num = DPO(front_overall, NS_normalize)
#         MOSA_DPO_num = DPO(front_overall, MOSA_normalize)
#         # 计算比例
#         NSGA3_DPO = NSGA3_DPO_num / len(NSGA3_normalize)
#         NSGA2_DPO = NSGA2_DPO_num / len(NSGA2_normalize)
#         NS_DPO = NS_DPO_num / len(NS_normalize)
#         MOSA_DPO = MOSA_DPO_num / len(MOSA_normalize)

#         '''记录各项评估指标'''
#         algo_results[0] += [len(NSGA3_normalize), NSGA3_time, NSGA3_IGD, NSGA3_DPO, NSGA3_volume] # 每算完一个特征划分方案就在每种算法这一行后边加上前沿解数量，计算时间，以及IGD，DPO，HV三个指标的值
#         algo_results[1] += [len(NSGA2_normalize), NSGA2_time, NSGA2_IGD, NSGA2_DPO, NSGA2_volume]
#         algo_results[2] += [len(MOSA_normalize), MOSA_time, MOSA_IGD, MOSA_DPO, MOSA_volume]
#         algo_results[3] += [len(NS_normalize), NS_time, NS_IGD, NS_DPO, NS_volume]

#     '''保存各次计算的评估指标'''
#     with open(os.path.join('results_all', 'indicator_test' + str(i) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
#         writer = csv.writer(f)
#         writer.writerows(algo_results)


""""""
"""单次计算单独观察各项评估指标"""
""""""
# # 要评估的加工特征序列
# feature_solution_num = 2

# # 测试次数编号
# test_num = 1

# """最佳前沿观察"""
# '''读取多目标函数值数据'''
# # NSGAIII_1000
# NSGA_path = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NSGAIII_600_' + str(test_num) + '.csv')
# NSGA_data = front_read(NSGA_path)
# NSGA_objects_X = [a[-1] for a in NSGA_data]
# # NSGA_X, NSGA_objects_X, NSGA_SE_X = tidy(NSGA_data)
# # for i in NSGA_objects_X:
# #     for j in i:
# #         j += [1] # 为便于识别出是哪种模型的结果，在多目标函数值后边加上一位数

# # NSGAII_1000
# NSGA2_path = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NSGAII_600_' + str(test_num) + '.csv')
# NSGA2_data = front_read(NSGA2_path)
# NSGA2_objects_X = [a[-1] for a in NSGA2_data]

# # NS_1000
# NS_path = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NS_600_' + str(test_num) + '.csv')
# NS_data = front_read(NS_path)
# NS_objects_X = [a[-1] for a in NS_data]
# # NS_X, NS_objects_X, NS_SE_X = tidy(NS_data)
# # for i in NS_objects_X:
# #     for j in i:
# #         j += [2] # 为便于识别出是哪种模型的结果，在多目标函数值后边加上一位数

# # MOSA_1000
# MOSA_path = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_MOSA_600_' + str(test_num) + '.csv')
# MOSA_data = front_read(MOSA_path)
# MOSA_objects_X = [a[-1] for a in MOSA_data]
# # MOSA_X, MOSA_objects_X, NOSA_SE_X = tidy(MOSA_data)


# '''排序与保存'''
# # solutions = NSGA_X + NS_X
# # objects_calculation = NSGA_objects_X + NS_objects_X
# # SE_set = NSGA_SE_X + NS_SE_X
# # fronts = ENS_5(solutions, objects_calculation, SE_set)

# # # 整理一下前沿，合并在一起，便于保存
# # fronts_1 = []
# # NSGA_F_num = []
# # NS_F_num = []
# # for i in range(len(fronts)):
# #     c = 0
# #     d = 0
# #     for j in fronts[i]:
# #         a = j[0:3] + [j[-1][0:4]]
# #         fronts_1.append([i] + a + [j[-1][-1]])
# #         if j[-1][-1] == 1:
# #             c += 1
# #         else:
# #             d += 1
# #     NSGA_F_num.append(c)
# #     NS_F_num.append(d)

# # 保存文件
# # with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'comparison_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
# #     writer = csv.writer(f)
# #     writer.writerows(fronts_1)

# # print('NSGA_F_num:')
# # print(NSGA_F_num)
# # print('NS_F_num')
# # print(NS_F_num)


# '''超体积比较(越大越好)'''
# # 超体积计算
# min_NSGA, max_NSGA = min_max(NSGA_objects_X)
# min_NS, max_NS = min_max(NS_objects_X)
# # min_MOSA, max_MOSA = min_max(MOSA_objects_X)
# # min_overall = [min(min_NSGA[k], min_NS[k], min_MOSA[k]) for k in range(len(min_NSGA))]
# # max_overall = [max(max_NSGA[k], max_NS[k], max_MOSA[k]) for k in range(len(max_NSGA))]
# min_overall = [min(min_NSGA[k], min_NS[k]) for k in range(len(min_NSGA))]
# max_overall = [max(max_NSGA[k], max_NS[k]) for k in range(len(max_NSGA))]
# NSGA_normalize = object_normalize(NSGA_objects_X, min_overall, max_overall)
# NS_normalize = object_normalize(NS_objects_X, min_overall, max_overall)
# # MOSA_normalize = object_normalize(MOSA_objects_X, min_overall, max_overall)
# NSGA_volume = hypervolume(NSGA_normalize)
# NS_volume = hypervolume(NS_normalize)
# # MOSA_volume = hypervolume(MOSA_normalize)
# print('HV:')
# print(NSGA_volume)
# print(NS_volume)
# # print(MOSA_volume)


# '''与全局最优前沿的平均距离比较（越小越好）'''
# # 定义规范化处理函数
# def object_normalize_1(object_front, min_overall, max_overall): # 以最佳前沿的各目标值集合、所有前沿的各目标的最小值和最大值集合作为输入
#     object_front_nor = [[None for _ in range(len(object_front[0]))] for _ in range(len(object_front))]
#     for k in range(len(object_front[0])):
#         delta = max_overall[k] - min_overall[k]
#         for i in range(len(object_front)):
#             object_front_nor[i][k] = (object_front[i][k] - min_overall[k]) / delta # 所有目标值都用1减一下
#     return object_front_nor

# # 规范化
# NSGA_objects_nor = object_normalize_1(NSGA_objects_X, min_overall, max_overall)
# NS_objects_nor = object_normalize_1(NS_objects_X, min_overall, max_overall)
# # MOSA_objects_nor = object_normalize_1(MOSA_objects_X, min_overall, max_overall)

# # 获取全局最佳前沿
# # objects_nor_overall = NSGA_objects_nor + NS_objects_nor + MOSA_objects_nor
# objects_nor_overall = NSGA_objects_nor + NS_objects_nor
# front_overall = ENS_3(objects_nor_overall)

# # 求平均距离
# NSGA_IGD = IGD(front_overall, NSGA_objects_nor)
# NS_IGD = IGD(front_overall, NS_objects_nor)
# # MOSA_IGD = IGD(front_overall, MOSA_objects_nor)
# print('IGD:')
# print(NSGA_IGD)
# print(NS_IGD)
# # print(MOSA_IGD)


# '''分散性比较（越小越好）'''
# NSGA_ES = ES(NSGA_objects_nor)
# NS_ES = ES(NS_objects_nor)
# # MOSA_ES = ES(MOSA_objects_nor)
# print('ES:')
# print(NSGA_ES)
# print(NS_ES)
# # print(MOSA_ES)


# '''未被其他算法得到的前沿支配的元素的数量比较（越大越好）'''
# NSGA_DPO = DPO(front_overall, NSGA_objects_nor) # 与全局最优前沿比较和与其他方法获得的前沿的集合比较结果是一样的
# NS_DPO = DPO(front_overall, NS_objects_nor)
# # MOSA_DPO = DPO(front_overall, MOSA_objects_nor)
# print('DPO:')
# print(NSGA_DPO)
# print(NS_DPO)
# # print(MOSA_DPO)


""""""
"""合并计算不同加工特征划分方案的配置结果"""
""""""
# test_num_total = 4 # 一共四次计算
# feature_solution_num_total = 4 # 一共四种特征划分方案

# algo_overall_results = [[] for _ in range(4)] # 用于记录每个算法的总体评估指标
# algo_pareto_num = [[] for _ in range(4)] # 用于记录每个算法的前沿解数量分布
# for i in range(1, test_num_total+1):
#     '''读取多目标函数值数据'''
#     NSGA3_objects_all = [] # 用于将多个特征划分方案下得到的前沿合并
#     NSGA2_objects_all = []
#     NS_objects_all = []
#     MOSA_objects_all = []
#     NSGA3_distribution = [[] for _ in range(feature_solution_num_total)] # 用于分开存放前沿数据，便于之后统计解的分布
#     NSGA2_distribution = [[] for _ in range(feature_solution_num_total)]
#     NS_distribution = [[] for _ in range(feature_solution_num_total)]
#     MOSA_distribution = [[] for _ in range(feature_solution_num_total)]
#     for j in range(1, feature_solution_num_total+1):  
#         # NSGAIII
#         NSGA3_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_NSGAIII_600_' + str(i) + '.csv')
#         NSGA3_data = front_read(NSGA3_path)
#         NSGA3_objects_X = [a[-1] for a in NSGA3_data]
#         NSGA3_objects_all += NSGA3_objects_X
#         NSGA3_distribution[j-1] += NSGA3_objects_X
#         # NSGAII
#         NSGA2_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_NSGAII_600_' + str(i) + '.csv')
#         NSGA2_data = front_read(NSGA2_path)
#         NSGA2_objects_X = [a[-1] for a in NSGA2_data]
#         NSGA2_objects_all += NSGA2_objects_X
#         NSGA2_distribution[j-1] += NSGA2_objects_X
#         # NS
#         NS_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_NS_600_' + str(i) + '.csv')
#         NS_data = front_read(NS_path)
#         NS_objects_X = [a[-1] for a in NS_data]
#         NS_objects_all += NS_objects_X
#         NS_distribution[j-1] += NS_objects_X
#         # MOSA
#         MOSA_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'best_front_MOSA_600_' + str(i) + '.csv')
#         MOSA_data = front_read(MOSA_path)
#         MOSA_objects_X = [a[-1] for a in MOSA_data]
#         MOSA_objects_all += MOSA_objects_X
#         MOSA_distribution[j-1] += MOSA_objects_X

#     '''获得各算法不同方案合并后的最佳前沿'''
#     NSGA3_front = ENS_3(NSGA3_objects_all)
#     NSGA2_front = ENS_3(NSGA2_objects_all)    
#     NS_front = ENS_3(NS_objects_all)
#     MOSA_front = ENS_3(MOSA_objects_all)

#     '''超体积比较(越大越好)'''
#     # 求各优化目标方向上的全局极值
#     min_NSGA3, max_NSGA3 = min_max(NSGA3_front)
#     min_NSGA2, max_NSGA2 = min_max(NSGA2_front)
#     min_NS, max_NS = min_max(NS_front)
#     min_MOSA, max_MOSA = min_max(MOSA_front)
#     min_overall = [min(min_NSGA3[k], min_NSGA2[k], min_NS[k], min_MOSA[k]) for k in range(len(min_NSGA3))]
#     max_overall = [max(max_NSGA3[k], max_NSGA2[k], max_NS[k], max_MOSA[k]) for k in range(len(max_NSGA3))]
#     # 规范化
#     NSGA3_normalize = object_normalize(NSGA3_front, min_overall, max_overall)
#     NSGA2_normalize = object_normalize(NSGA2_front, min_overall, max_overall)
#     NS_normalize = object_normalize(NS_front, min_overall, max_overall)
#     MOSA_normalize = object_normalize(MOSA_front, min_overall, max_overall)
#     # 大小反转（原本各指标越小越好，但为便于求超体积，这里反转成越大越好）
#     NSGA3_reverse = object_reverse(NSGA3_normalize)
#     NSGA2_reverse = object_reverse(NSGA2_normalize)
#     NS_reverse = object_reverse(NS_normalize)
#     MOSA_reverse = object_reverse(MOSA_normalize)
#     # 求超体积
#     NSGA3_volume = hypervolume(NSGA3_reverse)
#     NSGA2_volume = hypervolume(NSGA2_reverse)
#     NS_volume = hypervolume(NS_reverse)
#     MOSA_volume = hypervolume(MOSA_reverse)

#     '''与全局最优前沿的平均距离比较（越小越好）'''
#     # 获取规范化后的全局最佳前沿
#     front_all = NSGA3_normalize + NSGA2_normalize + NS_normalize + MOSA_normalize
#     front_best_overall = ENS_3(front_all)
#     # 求平均距离
#     NSGA3_IGD = IGD(front_best_overall, NSGA3_normalize)
#     NSGA2_IGD = IGD(front_best_overall, NSGA2_normalize)
#     NS_IGD = IGD(front_best_overall, NS_normalize)
#     MOSA_IGD = IGD(front_best_overall, MOSA_normalize)

#     '''未被其他算法得到的前沿支配的元素的比例比较（越大越好）'''
#     # 统计未被其他算法得到的前沿支配的元素的数量
#     NSGA3_DPO_num = DPO(front_best_overall, NSGA3_normalize)
#     NSGA2_DPO_num = DPO(front_best_overall, NSGA2_normalize)
#     NS_DPO_num = DPO(front_best_overall, NS_normalize)
#     MOSA_DPO_num = DPO(front_best_overall, MOSA_normalize)
#     # 计算比例
#     NSGA3_DPO = NSGA3_DPO_num / len(NSGA3_normalize)
#     NSGA2_DPO = NSGA2_DPO_num / len(NSGA2_normalize)
#     NS_DPO = NS_DPO_num / len(NS_normalize)
#     MOSA_DPO = MOSA_DPO_num / len(MOSA_normalize)

#     '''记录各项评估指标'''
#     algo_overall_results[0] += [len(NSGA3_normalize), NSGA3_IGD, NSGA3_DPO, NSGA3_volume] # 每算完一轮，就在每种算法这一行后边加上前沿解数量以及IGD，DPO，HV三个指标的值
#     algo_overall_results[1] += [len(NSGA2_normalize), NSGA2_IGD, NSGA2_DPO, NSGA2_volume]
#     algo_overall_results[2] += [len(MOSA_normalize), MOSA_IGD, MOSA_DPO, MOSA_volume]
#     algo_overall_results[3] += [len(NS_normalize), NS_IGD, NS_DPO, NS_volume]

#     '''统计各算法解各自合并后在各加工特征划分方案下的数量分布'''
#     # NSGAIII
#     NSGA3_pareto_num = []
#     for k in range(len(NSGA3_distribution)):
#         a = 0
#         for l in range(len(NSGA3_distribution[k])):
#             if NSGA3_distribution[k][l] in NSGA3_front:
#                 a += 1
#         NSGA3_pareto_num.append(a)
#     NSGA3_pareto_num.append(len(NSGA3_front))
#     # NSGAII
#     NSGA2_pareto_num = []
#     for k in range(len(NSGA2_distribution)):
#         a = 0
#         for l in range(len(NSGA2_distribution[k])):
#             if NSGA2_distribution[k][l] in NSGA2_front:
#                 a += 1
#         NSGA2_pareto_num.append(a)
#     NSGA2_pareto_num.append(len(NSGA2_front))
#     # NS
#     NS_pareto_num = []
#     for k in range(len(NS_distribution)):
#         a = 0
#         for l in range(len(NS_distribution[k])):
#             if NS_distribution[k][l] in NS_front:
#                 a += 1
#         NS_pareto_num.append(a)
#     NS_pareto_num.append(len(NS_front))
#     # MOSA
#     MOSA_pareto_num = []
#     for k in range(len(MOSA_distribution)):
#         a = 0
#         for l in range(len(MOSA_distribution[k])):
#             if MOSA_distribution[k][l] in MOSA_front:
#                 a += 1
#         MOSA_pareto_num.append(a)
#     MOSA_pareto_num.append(len(MOSA_front))

#     '''记录不同方案下解的数量分布'''
#     algo_pareto_num[0] += NSGA3_pareto_num # 每算完一轮就在每种算法这一行后边加上不同方案下的解数量以及解的总数
#     algo_pareto_num[1] += NSGA2_pareto_num
#     algo_pareto_num[2] += MOSA_pareto_num
#     algo_pareto_num[3] += NS_pareto_num

# '''保存各次计算的评估指标'''
# with open(os.path.join('results_all', 'indicator_overall' + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
#     writer = csv.writer(f)
#     writer.writerows(algo_overall_results)

# '''保存不同方案下解的数量分布'''
# with open(os.path.join('results_all', 'distribution_overall' + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
#     writer = csv.writer(f)
#     writer.writerows(algo_pareto_num)


""""""
"""多次计算基于超体积的收敛性同时观察"""
""""""
test_num_total = 4 # 一共四次计算
feature_solution_num_total = 4 # 一共四种特征划分方案

for i in range(1, test_num_total+1):
    algo_hypervolume = [] # 用于记录不同算法的超体积
    for j in range(1, feature_solution_num_total+1):
        '''读取一系列迭代周期的优化目标数据'''
        # object_NSGAIII
        object_NSGA3_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'object_NSGAIII_600_' + str(i) + '.csv')
        object_NSGA3 = object_read(object_NSGA3_path)
        # object_NSGAII
        object_NSGA2_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'object_NSGAII_600_' + str(i) + '.csv')
        object_NSGA2 = object_read(object_NSGA2_path)
        # object_NS
        object_NS_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'object_NS_600_' + str(i) + '.csv')
        object_NS = object_read(object_NS_path)
        # object_MOSA
        object_MOSA_path = os.path.join('results_all', 'results_test' + str(i), 'feature_solution_' + str(j), 'object_MOSA_600_' + str(i) + '.csv')
        object_MOSA = object_read(object_MOSA_path)

        '''求各优化目标方向上的全局极值'''
        objects_overall = object_NSGA3 + object_NSGA2 + object_NS + object_MOSA
        min_ksi = []
        max_ksi = []
        for k in range(len(objects_overall)):
            min_ks, max_ks = min_max(objects_overall[k])
            min_ksi.append(min_ks)
            max_ksi.append(max_ks)
        min_overall = []
        max_overall = []
        for k in range(len(min_ksi[0])):
            min_overall.append(min([a[k] for a in min_ksi]))
            max_overall.append(max([a[k] for a in max_ksi]))

        '''规范化处理'''
        # NSGAIII_normalizes
        NSGA3_normalizes = []
        for k in range(len(object_NSGA3)):
            NSGA3_normalizes.append(object_normalize(object_NSGA3[k], min_overall, max_overall))
        # NSGAII_normalizes
        NSGA2_normalizes = []
        for k in range(len(object_NSGA2)):
            NSGA2_normalizes.append(object_normalize(object_NSGA2[k], min_overall, max_overall))
        # NS_normalizes
        NS_normalizes = []
        for k in range(len(object_NS)):
            NS_normalizes.append(object_normalize(object_NS[k], min_overall, max_overall))
        # MOSA_normalizes
        MOSA_normalizes = []
        for k in range(len(object_MOSA)):
            MOSA_normalizes.append(object_normalize(object_MOSA[k], min_overall, max_overall))

        '''大小反转（原本各指标越小越好，但为便于求超体积，这里反转成越大越好）'''
        # NSGAIII_reverses
        NSGA3_reverses = []
        for k in range(len(NSGA3_normalizes)):
            NSGA3_reverses.append(object_reverse(NSGA3_normalizes[k]))
        # NSGAII_reverses
        NSGA2_reverses = []
        for k in range(len(NSGA2_normalizes)):
            NSGA2_reverses.append(object_reverse(NSGA2_normalizes[k]))
        # NS_reverses
        NS_reverses = []
        for k in range(len(NS_normalizes)):
            NS_reverses.append(object_reverse(NS_normalizes[k]))
        # MOSA_reverses
        MOSA_reverses = []
        for k in range(len(MOSA_normalizes)):
            MOSA_reverses.append(object_reverse(MOSA_normalizes[k]))

        '''超体积计算'''
        # NSGA3_volumes
        print('NSGAIII_volumes:')
        NSGA3_volumes = []
        for k in range(len(NSGA3_reverses)):
            NSGA3_volumes.append(hypervolume(NSGA3_reverses[k]))
            progress_bar(k, len(NSGA3_reverses))
        # NSGA2_volumes
        print('NSGAII_volumes:')
        NSGA2_volumes = []
        for k in range(len(NSGA2_reverses)):
            NSGA2_volumes.append(hypervolume(NSGA2_reverses[k]))
            progress_bar(k, len(NSGA2_reverses))
        # NS_volumes
        print('NS_volumes:')
        NS_volumes = []
        for k in range(len(NS_reverses)):
            NS_volumes.append(hypervolume(NS_reverses[k]))
            progress_bar(k, len(NS_reverses))
        # MOSA_volumes
        print('MOSA_volumes:')
        MOSA_volumes = []
        for k in range(len(MOSA_reverses)):
            MOSA_volumes.append(hypervolume(MOSA_reverses[k]))
            progress_bar(k, len(MOSA_reverses))

        '''记录超体积数据'''
        algo_hypervolume.append(NSGA3_volumes)
        algo_hypervolume.append(NSGA2_volumes)
        algo_hypervolume.append(MOSA_volumes)
        algo_hypervolume.append(NS_volumes)

    '''保存超体积数据'''
    algo_hypervolume = list(map(list, zip(*algo_hypervolume))) # 将结果的行列对换，每一列表示一种算法的一系列超体积值
    with open(os.path.join('results_all', 'hypervolume_test' + str(i) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(algo_hypervolume)


""""""
"""单次计算基于超体积的收敛性单独观察"""
""""""
# '''读取数据并显示图像'''
# ### object_NSGAIII ###
# object_path = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NSGAIII_600_' + str(test_num) + '.csv')
# results = object_read(object_path)

# # 记录一下全局最优前沿
# front_best = copy.deepcopy(results[0])
# front_bests = [front_best]
# # for i in range(1, 3):
# for i in range(1, len(results)):
#     front_best = ENS_3(results[i] + front_best) # 更新一下全局最优前沿
#     front_bests.append(copy.deepcopy(front_best))
#     progress_bar(i, len(results))
# objects_NSGA = front_bests


# ### object_NSGAII ###
# object_path = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NSGAII_600_' + str(test_num) + '.csv')
# results = object_read(object_path)

# # 记录一下全局最优前沿
# front_best = copy.deepcopy(results[0])
# front_bests = [front_best]
# # for i in range(1, 3):
# for i in range(1, len(results)):
#     front_best = ENS_3(results[i] + front_best) # 更新一下全局最优前沿
#     front_bests.append(copy.deepcopy(front_best))
#     progress_bar(i, len(results))
# objects_NSGA2 = front_bests


# ### object_NS ###
# # # 前400个迭代周期结果
# # object_path1 = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NSGAIII_600_' + str(test_num) + '.csv')
# # results1 = object_read(object_path1)

# # # 记录一下全局最优前沿
# # front_best = copy.deepcopy(results1[0])
# # front_bests = [front_best]
# # # for i in range(1, 3):
# # for i in range(1, 401):
# #     front_best = ENS_3(results1[i] + front_best) # 更新一下全局最优前沿
# #     front_bests.append(copy.deepcopy(front_best))
# #     progress_bar(i, 401)

# # # 后600个迭代周期结果
# # object_path2 = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NS_600_' + str(test_num) + '.csv')
# # results2 = object_read(object_path2)

# # objects_NS = front_bests + results2

# object_path2 = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NS_600_' + str(test_num) + '.csv')
# results2 = object_read(object_path2)

# # 记录一下全局最优前沿
# front_best = copy.deepcopy(results2[0])
# front_bests = [front_best]
# # for i in range(1, 3):
# for i in range(1, len(results2)):
#     front_best = ENS_3(results2[i] + front_best) # 更新一下全局最优前沿
#     front_bests.append(copy.deepcopy(front_best))
#     progress_bar(i, len(results))
# objects_NS = front_bests

# # ### object_MOSA ###
# # # 前400个迭代周期结果
# # object_path3 = os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_MOSA_600_' + str(test_num) + '.csv')
# # objects_MOSA = object_read(object_path3)




# ### 规范化处理 ###
# # objects_overall = objects_NSGA + objects_NS + objects_MOSA
# # objects_overall = objects_NSGA + objects_NSGA2
# objects_overall = objects_NSGA + objects_NSGA2 + objects_NS
# min_ksi = []
# max_ksi = []
# for i in range(len(objects_overall)):
#     min_ks, max_ks = min_max(objects_overall[i])
#     min_ksi.append(min_ks)
#     max_ksi.append(max_ks)
# min_overall = []
# max_overall = []
# for k in range(len(min_ksi[0])):
#     min_overall.append(min([a[k] for a in min_ksi]))
#     max_overall.append(max([a[k] for a in max_ksi]))

# NSGA_normalizes = []
# for i in range(len(objects_NSGA)):
#     NSGA_normalizes.append(object_normalize(objects_NSGA[i], min_overall, max_overall))

# NSGA2_normalizes = []
# for i in range(len(objects_NSGA2)):
#     NSGA2_normalizes.append(object_normalize(objects_NSGA2[i], min_overall, max_overall))

# NS_normalizes = []
# for i in range(len(objects_NS)):
#     NS_normalizes.append(object_normalize(objects_NS[i], min_overall, max_overall))

# # MOSA_normalizes = []
# # for i in range(len(objects_MOSA)):
# #     MOSA_normalizes.append(object_normalize(objects_MOSA[i], min_overall, max_overall))


# ### 超体积计算 ###
# NSGA_volumes = []
# for i in range(len(NSGA_normalizes)):
#     # 求超体积
#     NSGA_volumes.append(hypervolume(NSGA_normalizes[i]))
#     # print(volumes)
#     progress_bar(i, len(NSGA_normalizes))

# NSGA2_volumes = []
# for i in range(len(NSGA2_normalizes)):
#     # 求超体积
#     NSGA2_volumes.append(hypervolume(NSGA2_normalizes[i]))
#     # print(volumes)
#     progress_bar(i, len(NSGA2_normalizes))

# NS_volumes = []
# for i in range(len(NS_normalizes)):
#     # 求超体积
#     NS_volumes.append(hypervolume(NS_normalizes[i]))
#     # print(volumes)
#     progress_bar(i, len(NS_normalizes))

# # MOSA_volumes = []
# # for i in range(len(MOSA_normalizes)):
# #     # 求超体积
# #     MOSA_volumes.append(hypervolume(MOSA_normalizes[i]))
# #     # print(volumes)
# #     progress_bar(i, len(MOSA_normalizes))
    

# # ### 数据保存 ###
# # NSGA_volumes1 = NSGA_volumes.copy()
# # NS_volumes1 = NS_volumes.copy()
# # MOSA_volumes1 = MOSA_volumes.copy()

# # serie_length = max(len(NSGA_volumes), len(NS_volumes), len(MOSA_volumes))
# # if len(NSGA_volumes1) < serie_length:
# #     for _ in range(len(NSGA_volumes1), serie_length):
# #         NSGA_volumes1.append(0)
# # if len(NS_volumes1) < serie_length:
# #     for _ in range(len(NS_volumes1), serie_length):
# #         NS_volumes1.append(0)
# # if len(MOSA_volumes1) < serie_length:
# #     for _ in range(len(MOSA_volumes1), serie_length):
# #         MOSA_volumes1.append(0)

# # volumes_serie = []
# # for i in range(serie_length):
# #     volumes_serie.append([NSGA_volumes1[i], NS_volumes1[i], MOSA_volumes1[i]])

# # with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'hypervolume_plots_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
# #     writer = csv.writer(f)
# #     writer.writerows(volumes_serie)


# ### 图像显示 ###
# plt.plot(NSGA_volumes, color='r') # NSGAIII
# plt.plot(NSGA2_volumes, color='y') # NSGAII
# plt.plot(NS_volumes,color='b') # NS
# # plt.plot(MOSA_volumes, color='black')
# plt.show()




# 统计最佳前沿数量
# result_num = []
# for i in range(len(results)):
#     result_num.append(len(results[i]))
# plt.plot(result_num)
# plt.show()

# 统计每次最佳前沿中新增的元素
# new_num = []
# for i in range(1,len(results)):
#     new_num.append(len(list(filter(lambda x: x not in results[i-1], results[i]))))

# plt.plot(new_num)
# plt.show()

# 查看是否有前边支配后边的，以及有多少后边支配前边的
# for i in range(len(results)-100, len(results)):
#     a = 0
#     b = 0
#     for j in range(len(results[i])):
#         for k in range(len(results[i-1])):
#             if results[i][j][0] >= results[i-1][k][0] and results[i][j][1] >= results[i-1][k][1] and results[i][j][2] >= results[i-1][k][2] and results[i][j][3] >= results[i-1][k][3] and results[i][j] != results[i-1][k]:
#                 a += 1
#             if results[i][j][0] <= results[i-1][k][0] and results[i][j][1] <= results[i-1][k][1] and results[i][j][2] <= results[i-1][k][2] and results[i][j][3] <= results[i-1][k][3] and results[i][j] != results[i-1][k]:
#                 b += 1
#     print(a)
#     print(b)



# print(front_bests[6])

# 显示全局最优前沿元素数量变化
# front_num = []
# for a in front_bests:
#     front_num.append(len(a))
# plt.plot(a)
# plt.show()

# for i in range(1, len(front_bests)):
#     a = 0
#     b = 0
#     for j in range(len(front_bests[i])):
#         for k in range(len(front_bests[i-1])):
#             if front_bests[i][j][0] >= front_bests[i-1][k][0] and front_bests[i][j][1] >= front_bests[i-1][k][1] and front_bests[i][j][2] >= front_bests[i-1][k][2] and front_bests[i][j][3] >= front_bests[i-1][k][3] and front_bests[i][j] != front_bests[i-1][k]:
#                 a += 1
#             if front_bests[i][j][0] <= front_bests[i-1][k][0] and front_bests[i][j][1] <= front_bests[i-1][k][1] and front_bests[i][j][2] <= front_bests[i-1][k][2] and front_bests[i][j][3] <= front_bests[i-1][k][3] and front_bests[i][j] != front_bests[i-1][k]:
#                 b += 1
#     print(a)
#     print(b)



# # 统计最佳前沿数量
# result_num = []
# for i in range(len(front_bests)):
#     result_num.append(len(front_bests[i]))
# plt.plot(result_num)
# plt.show()

# 统计每次最佳前沿中新增的元素
# new_num = []
# for i in range(1,len(results)):
#     new_num.append(len(list(filter(lambda x: x not in results[i-1], results[i]))))

# plt.plot(new_num)
# plt.show()



# 查看未被下一轮迭代结果支配的结果数量


# 在终端输入 python result_observe.py 运行本程序