import os
import math
import time
import random
import numpy as np
import pandas as pd
from adaptation import adapt
from Cluster import cluster, first_edge, refer_point


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


# 计算两个解之间的Domination值
def dom(object_a, object_b):
    domination = 1
    for i in range(len(object_a)):
        if object_a[i] != object_b[i]:
            domination *= (2*abs(object_a[i]-object_b[i])/(object_a[i]+object_b[i]))
    return domination


# 邻域解生成算法
# 对一个个体的每一个工序随机选取一个可行的解
def perturb(p):
    rand = random.randint(0, len(p)-1)
    # print(p, rand)
    set_rand = quinary[quinary['J Feature'] == p[rand][1]]
    set_rand = set_rand.drop(set_rand[set_rand['quinary combination index'] == p[rand][0]].index)
    p[rand] = set_rand.sample(1).values[0].astype(int).tolist()[0:5]
    # for i in range(len(p)):
    #    set_i = quinary[quinary['J Feature'] == p[i][1]]
    #    p[i] = set_i.sample(1).values[0].astype(int).tolist()[0:5]
    new_p = p
    # print(new_p)
    return new_p


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


"""AMOSA算法参数设置"""
num_adap = 3  # 目标函数数量
t_initial = 500  # 初始温度
iter_max = 200  # 最大迭代次数
lowest_t = 0.1  # 最低温度
afa = 0.95  # 冷却速率
max_change = 100  # 当连续多次都不接受新的状态，开始改变温度
max_iter = 100  # 最大迭代次数
max_run = 10  # 最大运行次数
HL = 90  # 终止时归档集的大小
SL = 100  # 在使用聚类算法将其大小减少到HL之前，归档集大小可以达到的最大值
result_dir = '../result'
AMOSA_dir = 'AMOSA'
run_time = []  # 记录每次迭代的时间
for run in range(1, max_run+1):
    print('第', run, '次运行:')
    t3 = time.time()
    print('generate initialing......')
    '''生成初始解集并计算适应度'''
    temp_initials = pd.read_csv(os.path.join('../database', 'initials', 'initials_'+str(run)+'.csv')).values.tolist()
    initials = []
    for temp_initial in temp_initials:
        initials.append(location_to_process(quinary, temp_initial))
    print('initials:', initials)
    adaption_total = []
    for initial in initials:
        adaption_total.append(adapt(initial))
    t4 = time.time()
    print('genarate initialing is finished in ' + str(t4 - t3) + 's.')
    t5 = time.time()
    print('configuration optimizing......')
    '''AMOSA算法进行迭代'''
    edge_result = []  # 记录每次迭代的最优解
    Refer_point = refer_point(len(initials[0]), num_adap)  # 对某一特定维度的问题，参考点不变，可直接在外部定义
    archive = initials  # 归档集
    inx = random.randint(0, len(archive)-1)
    current_p = archive[inx]  # 在归档集中随机选取一个当前解
    temp = t_initial
    it = 0
    while it < iter_max:
        for _ in range(max_iter):
            new_p = perturb(current_p.copy())  # 生成邻域解作为新解
            adaption_current = adapt(current_p)  # 计算当前解适应度
            adaption_new = adapt(new_p)  # 计算新解的适应度
            dom_solution = []  # 记录归档集中支配新解的解的适应度
            anti_dom_solution = []  # 记录归档集中新解支配的解的位置
            anti_k = 0  # 记录归档集中新解支配的解的个数
            # 判断归档集与新解的支配关系,对目标函数较多的情况，通过比较差值来判断支配关系
            for person in range(len(adaption_total)):
                difference = np.array(adaption_new)-np.array(adaption_total[person])
                # print('single adaption:', adaption_new, adaption_total[person])
                # print('difference:', difference)
                if len(np.where(difference <= 0)[0]) == 0:  # 新解的目标函数均大于该解，该解支配新解
                    dom_solution.append(adaption_total[person])
                if len(np.where(difference > 0)[0]) == 0:  # 该解的目标函数均大于新解，新解支配该解
                    anti_dom_solution.append(person)
            k = len(dom_solution)  # 记录归档集中支配新解的解的个数
            anti_k = len(anti_dom_solution)  # 记录归档集中新解支配的解的个数
            dom_avg = 0
            for d in range(k):
                dom_avg += dom(adaption_new, dom_solution[d])
            difference = np.array(adaption_new) - np.array(adaption_current)  # 判断新解与旧解的支配关系
            # 第一种情况,新解的目标函数均大于当前解，当前解支配新解
            if len(np.where(difference <= 0)[0]) == 0:
                dom_avg_1 = (dom(adaption_current, adaption_new)+dom_avg)/(k+1)
                if random.random() < 1 / (1 + math.exp(dom_avg_1/temp)):  # 更新解集的概率
                    # print(1 / (1 + math.exp(dom_avg_1/temp)), 1 / (1 + math.exp(temp * dom_avg_1)))
                    current_p = new_p  # 以一定概率将新解设置为当前解
            # 第二种情况,新解与当前解互不支配
            elif len(np.where(difference <= 0)[0]) < len(difference):
                if k > 0:  # 归档集中存在支配新解的解
                    dom_avg_2 = dom_avg/k
                    if random.random() < 1/(1+math.exp(dom_avg_2/temp)):  # 更新解集的概率
                        # print('dom_avg_2:', dom_avg_2, 1/(1+math.exp(temp*dom_avg_2)))
                        current_p = new_p  # 以一定概率将新解设置为当前解
                if anti_k == 0 and k == 0:  # 归档集中没有支配新解的解,新解也不支配任何解，新解与归档集处于同一前沿
                    current_p = new_p  # 将新解设置为当前解
                    archive.append(new_p)
                    adaption_total.append(adaption_new)  # 将新解添加到归档集中
                    if len(archive) > SL:
                        print(len(archive))
                        archive, adaption_total = cluster(archive, Refer_point, adaption_total, HL)
                        print(len(archive))
                if anti_k > 0:  # 归档集中存在被新解支配的解
                    current_p = new_p  # 将新解设置为当前解
                    archive.append(new_p)
                    adaption_total.append(adaption_new)  # 将新解添加到归档集中
                    anti_dom_solution.reverse()  # 对索引进行反转，使其从后往前删除,避免元素移位
                    for remo in range(anti_k):  # 从归档集中删除被支配的解
                        archive.pop(anti_dom_solution[remo])
                        adaption_total.pop(anti_dom_solution[remo])
            # 第三种情况，当前解的目标函数均大于新解，新解支配当前解
            elif len(np.where(difference > 0)[0]) == 0:
                if k > 0:  # 新解又被归档集中其他解支配
                    dom_min = dom(adaption_new, dom_solution[0])
                    ort = 0  # dom值最小的位置,先设为位置0
                    for d in range(k):
                        if dom_min > dom(adaption_new, dom_solution[d]):
                            dom_min = dom(adaption_new, dom_solution[d])
                            ort = d
                    if random.random() < 1 / (1 + math.exp(-dom_min)):  # 更新解集的概率
                        current_p = archive[ort]  # 将当前解设为dom值最小的解
                    else:
                        current_p = new_p  # 将新解设置为当前解
                if anti_k >= 0:  # 新解支配归档集中某些解（当前解或其他解），当anti_k=0时当前解不在归档集中
                    current_p = new_p  # 将新解设置为当前解
                    archive.append(new_p)
                    adaption_total.append(adaption_new)  # 将新解添加到归档集中
                    anti_dom_solution.reverse()  # 对索引进行反转，使其从后往前删除,避免元素移位
                    for remo in range(anti_k):  # 从归档集中删除被支配的解
                        # print(anti_dom_solution[remo])
                        archive.pop(anti_dom_solution[remo])
                        adaption_total.pop(anti_dom_solution[remo])
        generate_edge, adaption_edge = first_edge(archive, adaption_total)
        edge_result.append(adaption_edge)
        temp = afa * temp  # 改变温度
        it += 1
        print('迭代次数:', it, '温度:', temp, '前沿解数量:', len(adaption_edge), '最优前沿:', adaption_edge)
    if len(archive) > SL:
        archive, adaption_total = cluster(archive, Refer_point, adaption_total, HL)
    # 保存每次迭代的前沿解为csv文件
    file = pd.DataFrame(edge_result)
    file.to_csv(os.path.join('../result', 'AMOSA', 'edge_result_' + str(run) + '.csv'), index=False, header=False)
    print(archive)
    print(adaption_total)
    t6 = time.time()
    run_time.append(t6 - t3)
    print('configuration optimizing is finished in ' + str(t6 - t3) + 's.')
pd.DataFrame(run_time).to_csv(os.path.join('../result', 'time', 'AMOSA.csv'), index=False, header=False)



