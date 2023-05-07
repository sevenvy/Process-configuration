"""多目标粒子群算法"""
import os
import sys
import math
import random
import time
import numpy as np
import pandas as pd
from adaptation import adapt
from Cluster import refer_point, last_selection


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


# 计算一个位置的适应度值
# 先将位置转化为工艺参数集，再计算对应工艺参数集的适应值
def location_adapt(location):
    process = location_to_process(quinary, location)
    adaption = adapt(process)
    return adaption


# 生成n个初始速度矢量
def v_generate(j, n):
    v = []
    for _ in range(n):
        v_1 = []
        for feature in range(len(j)):  # 对于每个加工特征分别随机选取五元组合
            v_1.append(v_initial)
        v.append(v_1)
    return v


def first_edge(archive, adaption):
    generate = np.array(archive)
    adap = np.array(adaption)
    size = len(generate)
    Np = np.zeros([size, 1], dtype=int)  # 记录每个个体的np值，即种群中支配该个体的个数
    Sp_all = []  # 记录每个个体的sp集合，即种群中被该个体支配的个体位置集合
    for p in range(size):  # 计算第p个个体的np、sp
        Sp = []
        for q in range(size):  # 遍历整个种群
            diff = adap[p] - adap[q]  # 对目标函数较多的情况，通过比较差值来判断支配关系
            if len(np.where(diff <= 0)[0]) == 0:  # p的目标函数均大于q
                Np[p] += 1  # q支配p，记录每个个体的Np值
            elif len(np.where(diff > 0)[0]) == 0:  # q的目标值均大于或等于p
                Sp.append(q)  # p支配q，Sp集合中记录的是个体的位置
        Sp_all.append(Sp)  # 每个个体的Sp集合
    inx, alnm = np.where(Np == 0)  # 找出种群中Np值为0的个体的索引
    choose_adap = adap[inx]
    choose_generate = generate[inx]
    adap_edge = choose_adap.tolist()
    generate_edge = choose_generate.tolist()
    return generate_edge, adap_edge


# 聚类算法,控制归档集数量
# 先通过快速非支配排序对归档集划分前沿，选择前HL个解
# 选择机制优化，原论文通过拥挤度排序，其在高维目标空间显然作用不太明显，这里考虑通过引入广泛分布参考点来维持集群的多样性
def cluster(archive, adaption, refer_point, num_reserve):
    """

    :param archive:需要进行聚类的解集
    :param adaption:输入解集的适应度集合
    :param refer_point:参考点，对某一特定维度的问题，参考点不变
    :param num_reserve:每次选择后保留的解集数量
    :return:经过选择后的解集以及每个parato前沿所包含的解的个数
    """

    generate = np.array(archive)
    adap = np.array(adaption)
    ideal_point = np.array(np.min(adap, axis=0)).reshape(1, len(adaption[0]))  # 求当前解集的理想点
    choose_generate = np.zeros([0, len(archive[0])], dtype=int)  # 非支配解总集合
    choose_adap = np.zeros([0, len(adaption[0])])  # 非支配解的适应度集合，与解一一对应
    size = len(generate)
    Np = np.zeros([size, 1], dtype=int)  # 记录每个个体的np值，即种群中支配该个体的个数
    Sp_all = []  # 记录每个个体的sp集合，即种群中被该个体支配的个体位置集合
    pareto_grade = []  # 记录每个等级的个体数量
    for p in range(size):  # 计算第p个个体的np、sp
        Sp = []
        for q in range(size):  # 遍历整个种群
            difference = adap[p] - adap[q]  # 对目标函数较多的情况，通过比较差值来判断支配关系
            if len(np.where(difference <= 0)[0]) == 0:  # p的目标函数均大于q
                Np[p] += 1  # q支配p，记录每个个体的Np值
            elif len(np.where(difference > 0)[0]) == 0:  # q的目标值均大于或等于p
                Sp.append(q)  # p支配q，Sp集合中记录的是个体的位置
        Sp_all.append(Sp)  # 每个个体的Sp集合
    if len(generate) <= num_reserve:
        inx, alum = np.where(Np == 0)  # 找出种群中Np值为0的个体的索引
        return generate[inx].tolist(), adap[inx].tolist()
    while len(choose_generate) < num_reserve:
        inx, alnm = np.where(Np == 0)  # 找出种群中Np值为0的个体的索引
        # 对这部分个体的Sp集合个体的np值减一
        for x in inx:
            for y in Sp_all[x]:
                Np[y] -= 1
        if len(choose_generate) + len(inx) <= num_reserve:
            choose_generate = np.append(choose_generate, generate[inx], axis=0)
            choose_adap = np.append(choose_adap, adap[inx], axis=0)
            pareto_grade.append(len(inx))
        else:
            i_index = last_selection(choose_adap, adap[inx], num_reserve - len(choose_generate), refer_point, ideal_point)
            i_generate = generate[inx][i_index]  # 在临界集合generate[inx]中选取i_index确定的解
            choose_generate = np.append(choose_generate, i_generate, axis=0)
            i_adap = adap[inx][i_index]  # 与i_generate对应
            choose_adap = np.append(choose_adap, i_adap, axis=0)
            pareto_grade.append(len(i_generate))
    result_set = choose_generate.tolist()
    result_adap = choose_adap.tolist()
    return result_set, result_adap


# 对外部档案集全局最优解进行选择，对同一前沿的解进行拥挤度排序，选择密度最低的解
def gbest_choose(rank, rank_adapt):
    """
    rank:集合i
    num:已选择的个体数量与需要的数量之间的差值，需要从集合i中选出
    """
    # 先按某一目标函数的大小对集合进行排序，以计算拥挤度
    rank = np.array(rank)
    rank_adapt = np.array(rank_adapt)
    rank = rank[np.argsort(-rank_adapt[:, 0], axis=0)]
    rank_adapt = rank_adapt[np.argsort(-rank_adapt[:, 0], axis=0)]
    crow = np.ones([len(rank), 1])  # 存储集合i中各个个体的拥挤度
    crow[0], crow[len(rank) - 1] = 0, 0  # 避免选择头尾解作为全局最优
    # 对中间元素计算拥挤度
    for q in range(1, len(rank) - 1):
        for ap in range(len(rank_adapt[0])):
            crow[q] *= abs(rank_adapt[q + 1, ap] - rank_adapt[q - 1, ap])
    # 通过轮盘赌选择需要的个体,基于拥挤度大小确定选取概率
    crow = crow / np.sum(crow)
    crow = np.cumsum(crow)
    rand = np.random.rand()
    ort = 0
    for j in range(len(crow)):
        if rand < crow[j]:
            ort = j
            break
    # 选取ort个体
    crow_generate = rank[ort].tolist()
    crow_adapt = rank_adapt[ort].tolist()
    return crow_generate, crow_adapt


# 计算两个解之间的Domination值
def dom(object_a, object_b):
    domination = object_a[0]/object_b[0]
    for i in range(1, len(object_a)):
        domination *= (object_a[i]/object_b[i])
    return domination


def dom_avg(loca):
    res = []
    temp1, temp2, temp3 = 0, 0, 0
    for i in range(len(loca)):
        temp1 += loca[i][0]
        temp2 += loca[i][1]
        temp3 += loca[i][2]
    res.append(temp1/len(loca))
    res.append(temp2/len(loca))
    res.append(temp3/len(loca))
    return res


"""参数设置"""
N = 100  # 粒子群规模
c1 = 0.6  # 自身认知因子，粒子下一步动作来源于自身经验部分所占的权重，将粒子推向个体最优位置
c2 = 0.3  # 社会认知因子，下一步动作来源于其它粒子经验部分所占的权重，将粒子推向群体最优位置
t_initial = 500  # 初始温度
afa = 0.97  # 冷却速率
v_initial = 0  # 初始速度
v_max = 1.0  # 限制最大速度
iter_max = 300  # 最大迭代次数
nochange_max = 100  # 最大无变化次数
max_run = 20  # 最大运行次数
SL = 100  # archive的最大容量
result_dir = '../result'  # 结果保存路径
MOAPSO_dir = 'MOAPSO'  # MOPSO算法结果保存路径
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
"""多目标粒子群算法寻优"""
run_time = []  # 保存每次迭代的时间
for run in range(1, max_run+1):
    print('第' + str(run) + '次运行', '......')
    t3 = time.time()
    print('Configuration optimizing by MOAPSO......')
    omiga = 0.9  # 惯性权重，上一代粒子的速度对当代粒子的速度的影响
    edge_result = []  # 保存每次迭代的最优解
    # 初始化粒子群位置，在0-1之间随机生成，初始化粒子群速度，全部置0
    locations = pd.read_csv(
        os.path.join('../database', 'initials', 'initials_' + str(run) + '.csv')).values.tolist()
    v = v_generate(j, N)
    locations_adapt = []
    for location in locations:
        locations_adapt.append(location_adapt(location))
    # 初始化archive解集，保存每次迭代获得的前沿解集
    Refer_point = refer_point(len(locations[0]), len(locations_adapt[0]))  # 对某一特定维度的问题，参考点不变，可直接在外部定义
    archive, archive_adapt = first_edge(locations, locations_adapt)
    # 初始化粒子群的个体最优位置，直接是初始化的粒子群位置
    p_best = locations.copy()
    # 初始化粒子群的个体最优适应值
    p_best_adapt = locations_adapt.copy()

    it = 0  # 迭代次数
    nochange_flag = 0  # 记录archive解集不变的次数
    t_current = t_initial  # 当前温度
    while it < iter_max and nochange_flag < nochange_max:
        # 初始化粒子群的全局最优位置与适应值,这里是单一位置
        g_best, g_best_adapt = gbest_choose(archive, archive_adapt)
        omiga = omiga * math.exp(-1/t_current)  # 更新惯性权重
        # 更新粒子群的位置和速度
        for one in range(N):
            for process in range(len(locations[one])):
                v[one][process] = omiga * v[one][process] + c1 * random.random() * (p_best[one][process] - locations[one][process]) + c2 * random.random() * (g_best[process] - locations[one][process])
                # 更新粒子群的位置
                locations[one][process] += v[one][process]
                # 越界粒子处理
                if locations[one][process] > 1:
                    locations[one][process] = 1-0.01*random.random()
                elif locations[one][process] < 0:
                    locations[one][process] = 0.01*random.random()
            # 更新适应度值
            locations_adapt[one] = location_adapt(locations[one])
        # 更新archive解集
        print('第', it, '次迭代  ', '当前温度:', t_current, '惯性权重：', omiga, 'archive解集：', len(archive_adapt), '全局最优：', g_best_adapt)
        # 对新生成的解进行选择，保留前沿解
        archive_new,  archive_adapt_new = first_edge(locations, locations_adapt)
        # 对新生成的解去重，并与archive解集合并
        archive += archive_new
        archive_adapt += archive_adapt_new
        # 在archive解集中选择，保留前沿解,同时保证解集的最大容量
        archive, archive_adapt = cluster(archive, archive_adapt, Refer_point, SL)
        # archive, archive_adapt = first_edge(archive, archive_adapt)
        # 对archive解集进行选择，保证解集的最大容量
        # if len(archive) > SL:
        #     archive, archive_adapt = crowded_sort(archive, archive_adapt, SL)
        edge_result.append(archive_adapt)
        # print(p_best_adapt)
        # print(locations_adapt)
        # 更新粒子群的个体最优位置和适应值
        for one in range(N):
            difference = np.array(locations_adapt[one]) - np.array(p_best_adapt[one])
            # 新位置的适应值均小于个体最优适应值，新解支配最优解，更新个体最优位置和适应值
            if len(np.where(difference > 0)[0]) == 0:
                p_best[one] = locations[one]
                p_best_adapt[one] = locations_adapt[one]
            # 新位置的适应值均大于个体最优适应值，最优解支配新解，不更新
            elif len(np.where(difference < 0)[0]) == 0:
                pass
            # 新位置与个体最优位置互不支配，随机选择一个解作为个体最优位置
            else:
                dom_solution = dom(p_best_adapt[one], locations_adapt[one])
                # print(1 / (1 + math.exp(dom_solution / t_current)), math.exp(-dom_solution / t_current))
                if random.random() < 1 / (1 + math.exp(dom_solution / t_current)):
                    p_best[one] = locations[one]
                    p_best_adapt[one] = locations_adapt[one]
        it += 1
        t_current = t_current * afa  # 更新温度
    file = pd.DataFrame(edge_result)
    file.to_csv(os.path.join(result_dir, MOAPSO_dir, 'edge_result_' + str(run) + '.csv'), index=False, header=False)
    t4 = time.time()
    run_time.append(t4 - t3)
    print('Configuration optimizing by MOAPSO is finished in ' + str(t4 - t3) + 's.')
pd.DataFrame(run_time).to_csv(os.path.join(result_dir, 'time', 'MOAPSO.csv'), index=False, header=False)
