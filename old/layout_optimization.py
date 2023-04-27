import random
import time
from collections import deque
import math
import matplotlib.pyplot as plt

from multi_process import multi_process_exec, multi_thread_exec

if __name__=='__main__':
    # 参数设置，在这里用于测试
    NMP = 20 # 车间设备工位数量
    NM_n = [2, 2, 1, 1, 4, 3, 1, 1, 1, 1]
    m1_n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    m2_j = [1, 2, 3, 1, 4, 5, 6, 7, 8, 5, 10, 9, 2, 5, 5, 5, 6, 6]
    a_nr = [[1,4],[2,13],[3],[5],[6,10,14,15,16],[7,17,18],[8],[9],[12],[11]]

    # 算法参数设置
    T_size = 15 # 禁忌表的大小
    S_size = 20 # 候选集合的大小
    TG1 = 500 # 总体最大迭代次数
    TG2 = 250 # 历史最优解持续未变化的最大迭代次数


'''每两个设备工位间的运输距离计算'''
# NMP 车间设备工位数量
def DMP_calculation(NMP):
    # 算法参数设置
    H = 4 # 行数
    L = 5 # 列数
    H_distance = 5 # 每两个横向相邻的设备工位间的间距
    L_distance = 5 # 每两个列向相邻的设备工位间的间距

    # 坐标分配
    locations = []
    for num in range(NMP):
        a = num // L
        b = num % L
        locations.append([a, b])

    # 求距离
    DMP = []
    for c in range(NMP): # 两两之间距离正向反向都算一下，便于后边计算运输距离时直接从中取值
        DMP1 = []
        for d in range(NMP):
            D = abs(locations[d][0] - locations[c][0]) * L_distance + abs(locations[d][1] - locations[c][1]) * H_distance # 采用曼哈顿距离
            DMP1.append(D)
        DMP.append(DMP1)

    return DMP


'''设计一个进度条，供后边参考使用'''  
def progress_bar(i, batches): # batches是进度条的总进度数量, i为循环标签
    a, b = '*' * math.floor((i + 1) / batches * 50), '.' * (50 - math.floor((i + 1) / batches * 50))
    c = i + 1
    # time.sleep(0.5)
    if c < batches:
        print('\r', "{}/".format(c), batches, "\t[{}{}]".format(a,b), sep ='', end = "", flush = True) # 前边持续刷新
    else:
        print('\r', "{}/".format(c), batches, "\t[{}{}]".format(a,b), sep ='') # 等最后一个batch算完，之后就不再刷新当前进度条了           
    # print('\n')
    # time.sleep(0.5)


'''禁忌搜索算法'''
# NM_n, NMP, DMP, m1_n, m2_j
# NM_n 不同型号设备的数量；NMP 车间设备工位数量；DMP 每两个设备工位间的运输距离；
# m1_n 所选设备中不同型号的编号；m2_j 各加工特征所选的设备编号
# PL 车间所有设备工位的编号
def layout_optimization(NM_n, NMP, DMP, m2_j, a_nr, T_size, S_size, TG1, TG2, PL):   
    # if sum(NM_n) > NMP:
    #     print('Number of machines is too large.')
    #     exit(0)
    
    # 单件产品总运输距离计算函数
    def DS_calculation(DMP, P_j, x):
        DS = 0
        for j in range(len(m2_j)-1):
            if m2_j[j] != m2_j[j+1]:
                d_j = 0
                for q1 in P_j[j]:
                    a = x[q1]
                    for q2 in P_j[j+1]:
                        d_j += DMP[a][x[q2]]  # 编号从0开始
                DS += d_j / (len(P_j[j]) * len(P_j[j+1]))
        return DS

    # 每道工序中的各设备对应的布局解中的元素编号
    position_num = 0
    P_j = [[] for _ in range(len(m2_j))]
    for n in range(len(NM_n)):
        B = list(range(position_num, position_num+NM_n[n]))
        for A in a_nr[n]:
            P_j[A-1] = B
        position_num += NM_n[n]

    # 初始解
    x = random.sample(range(NMP), sum(NM_n)) # 编号从0开始
    x_best = x
    T = deque(maxlen=T_size)
    k2 = 0

    # 初始解下的运输距离
    DS_best = DS_calculation(DMP, P_j, x)

    # DS_best_set = [] # 用于记录每次的最佳运输距离

    stop = 0 # 是否中途由于无可选解而需要立即停止迭代的标志
    for k1 in range(TG1):
        # 得出x的邻域解空间的操作集NS
        NP = set(PL) - set(x) # 空的工位
        NS1 = [(1, a, b) for a in range(len(x)) for b in NP] # 第a台设备的工位由x[a]变更为b
        NS2 = [(0, a, b) for a in range(len(x)-1) for b in range(a,len(x))] # 第a台设备和第b台设备的工位x[a]和x[b]互换
        NS = NS1 + NS2

        k3 = 0 # 作为是否用重新生成的候选解集的操作集的标志，k3=0，重新代入；k3=1，不再循环
        while k3 == 0:
            # 从邻域解空间操作集中随机抽取候选解集的操作集S
            if len(NS) > S_size:
                S = random.sample(NS, S_size)
            elif len(NS) > 0: # 如果邻域解空间数量已经不足了，那就直接以邻域解空间所有操作解作为候选解集的操作集
                S = NS
            else: 
                print('Error with the layout solution space: NS = null.')
                stop = 1
                break
        
            # 得到候选解集中所有解S以及相应情况下的运输距离DS
            S_x = [[] for _ in range(len(S))]
            DS_x = [[] for _ in range(len(S))]
            for e in range(len(S)):
                # 将操作解转换成实际解
                x1 = x.copy()
                if S[e][0] == 1:
                    x1[S[e][1]] = S[e][2]
                else:
                    x1[S[e][1]], x1[S[e][2]] = x1[S[e][2]], x1[S[e][1]] # 交换两个元素位置
                S_x[e] = x1
                DS_x[e] = DS_calculation(DMP, P_j, x1) # 求运输距离DS

            # 对解进行判断
            DS_min = min(DS_x)
            if DS_min < DS_best:
                x_index = DS_x.index(DS_min)
                x = S_x[x_index]
                x_best = x
                DS_best = DS_min
                k2 = 0
                k3 = 1
            else:
                # 去除候选解集中的禁忌解
                k4 = 0 # 用于记录删掉多少个元素
                for e in range(len(S)):
                    o = e - k4 # 删掉元素后，删除位置之后的元素序号就都-1，因此需要做个调整
                    if S[o] in T:
                        del(S[o])
                        del(S_x[o])
                        del(DS_x[o])
                        k4 += 1
                
                if S_x != []:
                    x_index = DS_x.index(min(DS_x))
                    x = S_x[x_index]
                    k2 += 1
                    k3 = 1

                else:
                    # 去除解空间的操作集中的禁忌操作
                    NS = list(filter(lambda r: r not in T, NS))

        # progress_bar(k1, TG1)
        # if k2 >= TG2:
        #     print('')

        if k2 >= TG2 or stop == 1:
            break

        # 更新禁忌表，禁忌表中储存的是变更操作
        T.append(S[x_index])

        # DS_best_set.append(DS_best)

    return x_best, DS_best # x_best 为各台设备的工位编号，从0开始，其形式为[p1, p2, ……]，按照每个型号设备以及数量，依次对应
    # return x_best, DS_best, DS_best_set, k1


if __name__=='__main__':
    DMP = DMP_calculation(NMP)
    PL = list(range(NMP)) # 编号从0开始
    
    step_num = [] # 用于记录迭代次数

    t1 = time.time()
    for i in range(10):
        MP_n, DS_best = layout_optimization(NM_n, NMP, DMP, m2_j, a_nr, T_size, S_size, TG1, TG2, PL)

        # step_num.append(k1-300)
        # if k1>=399:
        #     step_num.append(k1)
        # else:
        #     step_num.append(k1-300)
        # plt.plot(DS_best_set)
        # plt.show()
    t2 = time.time()

    # print(step_num)

    print('Single thread method: ' + str(t2-t1) + 's.')


    # t3 = time.time()
    # args_mat = [[NM_n, NMP, DMP,m2_j, a_nr, T_size, S_size, TG1, TG2] for _ in range(600)]
    # results = multi_thread_exec(layout_optimization, args_mat, 10, desc='多线程')
    # t4 = time.time()
    # print('Multi thread method: ' + str(t4-t3) + 's.')
    # DS_best = []
    # for i in range(10):
    #     DS_best.append(min(results[i*10+j][1] for j in range(10)))

    # t5 = time.time()
    # args_mat = [[NM_n, NMP, DMP, m2_j, a_nr, T_size, S_size, TG1, TG2, PL] for _ in range(2780)]
    # results = multi_process_exec(layout_optimization, args_mat, 60, desc='多进程')
    # t6 = time.time()
    # print('Multi process method: ' + str(t6-t5) + 's.')
    # DS_best = []
    # for i in range(10):
    #     DS_best.append(min(results[i*3+j][1] for j in range(3)))

    # # print(MP_n)
    # print(DS_best)

# 在终端输入 python layout_optimization.py 运行本程序