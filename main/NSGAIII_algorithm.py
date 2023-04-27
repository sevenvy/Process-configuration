import numpy as np
import random
import os
import time
import pandas as pd
from initial_generate import generate_initials
from adaptation import adapt
from Cluster import last_selection
from scipy.special import comb
from itertools import combinations


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


'''交叉与变异操作'''
def crossover_mutation(N, parents, quinary, pm):
    """
    :param N: 种群个体数量
    :param parents:父代
    :param quinary: 五元组合集合
    :param pm: 变异概率
    :return: childs 子代
    """
    childs = []
    # 交叉操作
    # 对父代个体进行随机两两分组，每组进行一次交叉形成两个新的子代个体
    parents = np.array(parents)
    index = list(range(len(parents)))
    random.shuffle(index)  # 打乱父代个体及其编号的顺序，之后按顺序两两交叉，也相当于随机分组后交叉
    parents = parents[index].tolist()
    size = len(parents[0])
    for n in range(int(N/2)):  # 当子代种群个体数量达到设定总数时停止迭代，由于每次交叉会产生2个子代个体，因此为N/2
        cross_point = random.sample(list(range(size)), 1)[0] # 随机给定交叉点，此处父代中第cross_point个五元组合及其之前的五元组合保留，此后的五元组合则相互交换
        child1 = parents[2*n][:cross_point] + parents[2*n+1][cross_point:]
        child2 = parents[2*n+1][:cross_point] + parents[2*n][cross_point:]
        childs.append(child1)
        childs.append(child2)
    # 变异操作
    for n in range(N):
        for j in range(len(childs[n])):
            if random.random() < pm:  # 以一定概率进行变异操作
                quinary_j = quinary[quinary['J Feature'] == childs[n][j][1]]  # 得到该五元组合的可行解集合
                quinary_j.drop([childs[n][j][0]], axis=0)  # 去除原五元组合
                childs[n][j] = quinary_j.sample(1).values[0].astype(int).tolist()[0:5]  # 随机选取一个可行解作为变异后的解
    # 强制变异操作
    # 如果存在与父代或其他子代相同的个体，则强制其随机一个五元组合进行变异操作，并且保证无重复
    for n in range(N):
        D = childs.copy() # 复制一下，避免直接赋值情况下两者共用索引，从而导致使用remove或del时，将两者共用的索引删去（remove或del是删除元素的索引）
        D.remove(D[n]) # 去掉当前个体
        Dn = childs[n]
        k1 = 1 # 是否需要判断是否存在相同项的标志
        k2 = 0 # 是否进行了强制变异的标志
        while k1 == 1:
            if Dn in D: # 如果仍然存在相同项
                Dn = childs[n]
                a = random.sample((0, len(childs[n])-1), 1)[0]  # 随机选择个体中的一个五元组合进行变异
                quinary_d = quinary[quinary['J Feature'] == Dn[a][1]]  # 得到该五元组合的可行解集合
                quinary_d.drop([Dn[a][0]], axis=0)  # 去除原五元组合
                Dn[a] = quinary_d.sample(1).values[0].astype(int).tolist()[0:5]  # 随机选取一个可行解作为变异后的解
                k2 = 1
            else:
                k1 = 0
                if k2 == 1: # 如果进行了强制变异，则将变异后的子代个体更新
                    childs[n] = Dn
    return childs


'''选择函数，基于Pareto等级的快速支配排序以及参考点对个体的优劣进行多目标排序'''
def choose(initials, refer_point, adaption, num_reserve):
    """
    :param initials:需要进行选择的种群
    :param refer_point:参考点，对某一特定维度的问题，参考点不变
    :param adaption:输入解集的适应度集合
    :param num_reserve:每次选择后保留的解集数量
    :return:经过选择后的种群及其适应度以及最优前沿的个体位置
    """
    generate = np.array(initials)
    adap = np.array(adaption)
    nomal_point = np.array(np.min(adap, axis=0)).reshape(1, len(adaption[0]))  # 求当前解集的理想点
    choose_generate = np.zeros([0, len(initials[0]), len(initials[0][0])], dtype=int)  # 非支配解总集合
    choose_adap = np.zeros([0, len(adaption[0])])  # 非支配解的适应度集合，与解一一对应
    size = len(generate)
    Np = np.zeros([size, 1], dtype=int)  # 记录每个个体的np值，即种群中支配该个体的个数
    Sp_all = []  # 记录每个个体的sp集合，即种群中被该个体支配的个体位置集合
    index_set = []  # 记录每个等级的个体位置集合,不包括最后一等级
    for p in range(size):  # 计算第p个个体的np、sp
        Sp = []
        for q in range(size):  # 遍历整个种群
            difference = adap[p] - adap[q]  # 对目标函数较多的情况，通过比较差值来判断支配关系
            if len(np.where(difference <= 0)[0]) == 0:  # p的目标函数均大于q
                Np[p] += 1  # q支配p，记录每个个体的Np值
            elif len(np.where(difference > 0)[0]) == 0:  # q的目标值均大于或等于p
                Sp.append(q)  # p支配q，Sp集合中记录的是个体的位置
        Sp_all.append(Sp)  # 每个个体的Sp集合
    while len(choose_generate) < num_reserve:
        inx, alnm = np.where(Np == 0)  # 找出种群中Np值为0的个体的索引
        # 对这部分个体的Sp集合个体的np值减一
        for x in inx:
            for y in Sp_all[x]:
                Np[y] -= 1
        if len(choose_generate) + len(inx) <= num_reserve:
            choose_generate = np.append(choose_generate, generate[inx], axis=0)
            choose_adap = np.append(choose_adap, adap[inx], axis=0)
            index_set.append(inx)
        else:
            index_set.append(inx)  # 这里记录的前沿数量可能大于num_reserve
            i_index = last_selection(choose_adap, adap[inx], num_reserve - len(choose_generate), refer_point, nomal_point)
            i_generate = generate[inx][i_index]  # 在临界集合generate[inx]中选取i_index确定的解
            choose_generate = np.append(choose_generate, i_generate, axis=0)
            i_adap = adap[inx][i_index]  # 与i_generate对应
            choose_adap = np.append(choose_adap, i_adap, axis=0)
    edge_set = adap[index_set[0]]  # 前沿解集的适应度集合
    result_set = choose_generate.tolist()
    result_adap = choose_adap.tolist()
    edge_set = edge_set.tolist()
    return result_set, result_adap, edge_set


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

    # 各目标的标准化
    for n in range(len(S1)):
        for j in range(len(a)):
            S1[n][-1][j] *= a[j]

    return S1


'''参考点创建'''
# 参考点定义，Deb and Jain’s Method，产生双层超平面，减少参考点个数
def uniformpoint(N, M):
    """
    :param N: 所要生成的参考点个数
    :param M: 目标函数维度
    :return:
    """
    h1 = 1
    while comb(h1 + M-1, M-1) <= N:
        h1 = h1+1
    h1 = h1-1
    W = np.array(list(combinations(range(h1+M-1), M-1)))-np.tile(np.array(list(range(M-1))), (int(comb(h1+M-1, M-1)), 1))
    W = (np.hstack((W, h1+np.zeros((W.shape[0], 1))))-np.hstack((np.zeros((W.shape[0], 1)), W)))/h1
    if h1 < M:
        h2 = 0
        while comb(h1+M-1, M-1)+comb(h2+M-1, M-1) <= N:
            h2 = h2+1
        h2 = h2-1
        if h2 > 0:
            W2 = np.array(list(combinations(range(h2+M-1), M-1)))-np.tile(np.array(list(range(M-1))), (int(comb(h2+M-1, M-1)), 1))
            W2 = (np.hstack((W2, h2+np.zeros((W2.shape[0], 1))))-np.hstack((np.zeros((W2.shape[0], 1)), W2)))/h2
            W2 = W2/2+1/(2*M)
            W = np.vstack((W, W2))  # 按列合并
    W[W < 1e-6] = 1e-6
    N = W.shape[0]
    return W


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
            d.append(1 - np.dot(S1[i][-1], referpoint_set[j]) / (np.linalg.norm(S1[i][-1]) * np.linalg.norm(referpoint_set[j]))) # 此处采用的是余弦距离，可参考：https://blog.csdn.net/lucky_kai/article/details/89514868
        
        # 获取与每个个体的最小距离及相应的参考点
        nearest_d.append(min(d))
        nearest_point.append(referpoint_set[d.index(min(d))])

    return nearest_point, nearest_d
    

def adaption_calculate(initials):
    # 计算初始种群中各个个体的自适应值
    adaption = []
    for i in initials:
        adaption.append(adapt(i))
    return adaption


'''NSGA-III算法参数设置'''
N = 100  # 种群个体数量，需要为偶数以便交叉操作 160
pm = 0.05  # 变异概率
max_iter = 200  # 总体最大迭代次数
max_nochange = 200  # 最优前沿持续未变化的最大迭代次数
max_run = 10  # 最大运行次数
result_dir = '../result'
NSGAIII_dir = 'NSGAIII'


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
run_time = []  # 记录每次迭代的时间
for run in range(1, max_run+1):
    print('第' + str(run) + '次运行', '......')
    t3 = time.time()
    print('Configuration optimizing by NSGA-III ......')
    '''初始种群生成'''
    temp_initials = pd.read_csv(os.path.join('../database', 'initials', 'initials_' + str(run) + '.csv')).values.tolist()
    initials = []
    for temp_initial in temp_initials:
        initials.append(location_to_process(quinary, temp_initial))
    print('initials:', initials)
    adaption_all = adaption_calculate(initials)  # 计算初始种群中各个个体的适应值
    refer_point = uniformpoint(len(initials[0]), len(adaption_all[0]))  # 对某一特定维度的问题，参考点不变，可直接在外部定义
    """NSGA-III算法寻优"""
    parents = initials
    adaption_parents = adaption_all
    edge_result = []
    iter = 0 # 迭代次数
    nochange_flag = 0 # 最优前沿持续未变化的次数
    while iter < max_iter and nochange_flag <= max_nochange:
        childs = crossover_mutation(N, parents, quinary, pm)  # 交叉与变异操作
        adaption_childs = adaption_calculate(childs)  # 计算子代中各个个体的自适应值
        P_C = parents + childs
        adaption_all = adaption_parents + adaption_childs
        iter += 1
        new_set, new_adaption, edge_set = choose(P_C, refer_point, adaption_all, N)   # 选择操作
        edge_result.append(edge_set)
        print('第', iter, '次迭代前沿：', edge_set)
        if new_set == parents:  # 判断种群是否发生变化
            nochange_flag += 1
        else:
            nochange_flag = 0
            parents = new_set
            adaption_parents = new_adaption
    file = pd.DataFrame(edge_result)
    file.to_csv(os.path.join(result_dir, NSGAIII_dir, 'edge_result_'+str(run)+'.csv'), index=False, header=False)
    t4 = time.time()
    run_time.append(t4 - t3)
    print('Configuration optimizing by NSGA-III is finished in ' + str(t4 - t3) + 's.')
pd.DataFrame(run_time).to_csv(os.path.join(result_dir, 'time', 'NSGAIII.csv'), index=False, header=False)



