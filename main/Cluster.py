import numpy as np
from scipy.special import comb
from itertools import combinations
import copy
import math
import pandas as pd
from adaptation import adapt
from initial_generate import generate_initials


# 参考点定义，Deb and Jain’s Method，产生双层超平面，减少参考点个数
def refer_point(N, M):
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


# 求两个向量矩阵的余弦值,x的列数等于y的列数
def pdist(x, y):
    x0 = x.shape[0]
    y0 = y.shape[0]
    xmy = np.dot(x, y.T)  # x乘以y
    xm = np.array(np.sqrt(np.sum(x**2, 1))).reshape(x0, 1)
    ym = np.array(np.sqrt(np.sum(y**2, 1))).reshape(1, y0)
    xmmym = np.dot(xm, ym)
    cos = xmy/xmmym
    return cos


# 对最后一个边缘解进行选择
def last_selection(popfun1, popfun2, K, Z, Zmin):
    """

    :param popfun1: 已选择集合的适应度
    :param popfun2: 临界集合的适应度
    :param K: 离目标子代数量的差额
    :param Z: 参考点
    :param Zmin: 理想点
    :return: 在临界集合中选择个体的索引
    """
    # 选择最后一个front的解
    popfun = copy.deepcopy(np.vstack((popfun1, popfun2))) - np.tile(Zmin, (popfun1.shape[0] + popfun2.shape[0], 1))
    N, M = popfun.shape[0], popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]

    # 正则化
    extreme = np.zeros(M)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun / (np.tile(w[i, :], (N, 1))), 1))

    # 计算截距
    extreme = extreme.astype(int)  # python中数据类型转换一定要用astype
    # temp = np.mat(popfun[extreme,:]).I
    temp = np.linalg.pinv(np.mat(popfun[extreme, :]))
    hyprtplane = np.array(np.dot(temp, np.ones((M, 1))))
    a = 1 / hyprtplane
    if np.sum(a == math.nan) != 0:
        a = np.max(popfun, 0)
    np.array(a).reshape(M, 1)  # 一维数组转二维数组
    # a = a.T - Zmin
    a = a.T
    popfun = popfun / (np.tile(a, (N, 1)))

    # 联系每一个解和对应向量
    # 计算每一个解最近的参考线的距离
    cos = pdist(popfun, Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun ** 2, 1))).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos ** 2)
    # 联系每一个解和对应的向量
    d = np.min(distance.T, 0)
    pi = np.argmin(distance.T, 0)

    # 计算z关联的个数
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)

    # 选出剩余的K个
    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        # 选择最不拥挤的参考点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
        #        I = np.ravel(np.array(np.where(choose == False)))
        #        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
        I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if I.shape[0] != 0:
            if rho[j] == 0:
                s = np.argmin(d[N1 + I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j] + 1
        else:
            zchoose[j] = False
    return choose


# 聚类算法,控制归档集数量
# 先通过快速非支配排序对归档集划分前沿，选择前HL个解
# 选择机制优化，原论文通过拥挤度排序，其在高维目标空间显然作用不太明显，这里考虑通过引入广泛分布参考点来维持集群的多样性
def cluster(archive, refer_point, adaption, num_reserve):
    """

    :param archive:需要进行聚类的解集
    :param refer_point:参考点，对某一特定维度的问题，参考点不变
    :param adaption:输入解集的适应度集合
    :param num_reserve:每次选择后保留的解集数量
    :return:经过选择后的解集以及每个parato前沿所包含的解的个数
    """
    generate = np.array(archive)
    adap = np.array(adaption)
    ideal_point = np.array(np.min(adap, axis=0)).reshape(1, len(adaption[0]))  # 求当前解集的理想点
    choose_generate = np.zeros([0, len(archive[0]), len(archive[0][0])], dtype=int)  # 非支配解总集合
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


def first_edge(archive, adaption):
    generate = np.array(archive)
    adap = np.array(adaption)
    size = len(generate)
    Np = np.zeros([size, 1], dtype=int)  # 记录每个个体的np值，即种群中支配该个体的个数
    Sp_all = []  # 记录每个个体的sp集合，即种群中被该个体支配的个体位置集合
    for p in range(size):  # 计算第p个个体的np、sp
        Sp = []
        for q in range(size):  # 遍历整个种群
            difference = adap[p] - adap[q]  # 对目标函数较多的情况，通过比较差值来判断支配关系
            if len(np.where(difference <= 0)[0]) == 0:  # p的目标函数均大于q
                Np[p] += 1  # q支配p，记录每个个体的Np值
            elif len(np.where(difference > 0)[0]) == 0:  # q的目标值均大于或等于p
                Sp.append(q)  # p支配q，Sp集合中记录的是个体的位置
        Sp_all.append(Sp)  # 每个个体的Sp集合
    inx, alnm = np.where(Np == 0)  # 找出种群中Np值为0的个体的索引
    choose_adap = adap[inx]
    choose_generate = generate[inx]
    adap_edge = choose_adap.tolist()
    generate_edge = choose_generate.tolist()
    return generate_edge, adap_edge


if __name__ == '__main__':
    num_process = 18
    num_adp = 3
    Z = (num_process, num_adp)  # 对某一特定维度的问题，参考点不变，可直接在外部定义
    j = pd.read_csv('../database/feature.csv')
    quinary = pd.read_csv('../database/quinary_combination.csv')
    generate = generate_initials(j, quinary, 100)
    adaption_total = []
    for initial in generate:
        adaption_total.append(adapt(initial))
    archive, adapti = cluster(generate, Z, adaption_total, 50)
    print(archive)
