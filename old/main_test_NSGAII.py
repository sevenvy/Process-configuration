import os
import math
import time
import csv
import copy


'''本程序仅在数值实验时使用'''

# 从其他python文件中调用函数
from quinary_combination import data_reading
from machine_number import NM_calculation
from layout_optimization import DMP_calculation, layout_optimization
from optimization_objects import variable_determine, CB_calculation, TB_calculation, CTP_calculation, TTP_calculation
from NSGAIII_algorithm import generate_initials, crossover_mutation, ENS, nomalization, refer_points, distance, last_selection, update, ENS_best
from NSGAII_algorithm import crowding_distance
from multi_process import multi_process_exec


if __name__ == '__main__': # 这里主要是用来避免多进程/线程时,反复重新导入数据

    '''参数设置'''
    # 直接设定：NP, TSH, TSH1, CPM, CAZ, CSV, CSV1, TVD, TVR, TV, CZJ, CD, CPR, NMP
    # 读取：J, M, T, F, AD, TJ, CJ
    # 由决策变量获得：m1_n, m2_j, t1_j, f1_j, a_nr, b_nks, c_nuv,
    # 决策变量：i1_j, NM_n, PO_jp
    # 在决策变量基础上进一步选择或优化计算：NM_n, DS

    # 单位制：小时，米，元

    # 配置模型参数设置
    NP = 200 # 待生产产品数量
    TSH = 0.2 # 安装一次刀具所需时间
    TSH1 = 0.05 # 刀具磨损更换一次所需时间
    CPM = 40 # 刀具安装或磨损更换的单位时间成本
    CAZ = 40 # 夹具安装的单位时间成本
    CSV = 30000 # 物料运输系统采购单价
    CSV1 = 1000 # 物料运输系统安装调试单价
    TVD = 360 # 物料运输系统的交货期
    TVR = 1 # 物料运输系统安装调试时间与加工设备数量间的比例系数
    TV = 1800 # 物料运输的平均运输速度
    CZJ = 40 # 工件装夹的单位时间成本
    CD = 0.003 # 单位运输距离的物料运输成本
    CPR_1 = 50 # 单件产品的原材料成本，纯机加工
    CPR_2 = 30 # 单件产品的原材料成本，3D打印
    NMP = 20 # 车间设备工位数量

    # 要评估的加工特征序列
    feature_solution_num = 4

    # 测试次数编号
    test_num = 4

    # 数据读取路径
    data_dir = 'database'
    feature_dir = 'feature'
    process_dir = 'process'
    machine_dir = 'machine'
    tool_dir = 'tool'
    fixture_dir = 'fixture'

    quinary_dir = 'quinary_combination'

    # NSGA-Ⅱ算法参数设置
    N = 50 # 种群个体数量，需要为偶数以便交叉操作 160
    pm = 0.05 # 变异概率
    object_num = 4 # 优化目标数量
    NG1 = 600 # 总体最大迭代次数
    NG2 = 200 # 最优种群持续未变化的最大迭代次数

    # 禁忌搜索算法参数设置
    T_size = 15 # 禁忌表的大小
    S_size = 20 # 候选集合的大小
    TG1 = 500 # 总体最大迭代次数 15000
    TG2 = 200 # 历史最优解持续未变化的最大迭代次数 5000
    RE = 1 # 重复迭代多少次求最小值

    # 计算线程/进程参数设置
    process_size = 60 # 布局优化进程数 # 不能超过60
    # process_size2 = 15 # 不同设备配置情况下的布局优化进程数


    '''数据读取'''
    t1 = time.time()
    print('Data loading ......')

    # 特征集
    J = data_reading(os.path.join(data_dir, feature_dir + '_' + str(feature_solution_num) + '.csv'))
    # 设备集
    M = data_reading(os.path.join(data_dir, machine_dir + '.csv'))
    # 刀具集
    T = data_reading(os.path.join(data_dir, tool_dir + '.csv'))
    # 夹具集
    F = data_reading(os.path.join(data_dir, fixture_dir + '.csv'))
    # 五元组合集
    Q = data_reading(os.path.join(quinary_dir, quinary_dir + str(feature_solution_num) + '.csv'))

    # 五元组合集整理
    Q_class = [[] for _ in range(len(J))]
    TJ = []
    CJ = []
    AD = []
    for i in range(len(Q)):
        Q_class[Q[i][0]-1].append([i] + Q[i][0:5]) # 按照特征进行过分类整理五元组合，在每个特征维度下，每个元素包含的参数为：五元组合编号（从0开始），特征J，工艺P，设备M，刀具T，夹具F
        TJ.append(Q[i][5]) # 特定五元组合下的工序加工时间
        CJ.append(Q[i][6]) # 特定五元组合下的工序加工成本
        AD.append(Q[i][7]) # 特定五元组合下的可选刀轴方向

    # t2 = time.time()
    # print('Data loading is finished in ' + str(t2 - t1) + 's.')


    '''设计一个进度条，供后边参考使用'''  
    def progress_bar(i, batches): # batches是进度条的总进度数量, i为循环标签
        length = 120 # 进度条长度
        a, b = '*' * math.floor((i + 1) / batches * length), '.' * (length - math.floor((i + 1) / batches * length))
        c = i + 1
        # time.sleep(0.5)
        if c < batches:
            print('\r', "{}/".format(c), batches, "\t[{}{}]".format(a,b), sep ='', end = "", flush = True) # 前边持续刷新
        else:
            print('\r', "{}/".format(c), batches, "\t[{}{}]".format(a,b), sep ='') # 等最后一个batch算完，之后就不再刷新当前进度条了           
        # print('\n')
        # time.sleep(0.5)


    '''全局配置'''
    # t3 = time.time()
    print('Initializing ......')

    # 每两个设备工位间的运输距离计算
    DMP = DMP_calculation(NMP)

    # 单件产品的原材料成本取值
    if 7 in [j[0] for j in J]: # 如果是3D打印的情况
        CPR = CPR_2
    else:
        CPR = CPR_1

    # 车间所有设备工位的编号
    PL = list(range(NMP)) # 编号从0开始

    # 生成初始种群
    initials, i_j = generate_initials(Q_class, N)

    # 定义一个函数用于计算目标函数值
    def objects_calculate(i_j):
        args_mat = []
        SE_num = [] # 用于记录每个个体下设备数量的配置方案数量
        SE_all = []
        m1_n_all, m2_j_all, t1_j_all, f1_j_all, a_nr_all, b_nks_all, c_nuv_all = [], [], [], [], [], [], []
        for i in range(len(i_j)):
            m1_n, m2_j, t1_j, f1_j, a_nr, b_nks, c_nuv = variable_determine(i_j[i], Q)
            SE = NM_calculation(NP, NMP, M, T, TSH1, TJ, AD, m1_n, m2_j, t1_j, f1_j, i_j[i], a_nr, b_nks) # 设备数量的可行解空间(不同型号设备的数量)
            SE_num.append(len(SE))
            SE_all.append(SE)
            m1_n_all.append(m1_n)
            m2_j_all.append(m2_j)
            t1_j_all.append(t1_j)
            f1_j_all.append(f1_j)
            a_nr_all.append(a_nr)
            b_nks_all.append(b_nks)
            c_nuv_all.append(c_nuv)
            for NM_n in SE:
                args_mat += [[NM_n, NMP, DMP, m2_j, a_nr, T_size, S_size, TG1, TG2, PL] for _ in range(RE)] # 将所有的可设置并行的布局优化过程中禁忌搜索过程的参数输入整理在一起
        # t61 = time.time()
        # print('solution space calculation: ' + str(t61-t6) + 's.')

        # 多进程并行计算
        results = multi_process_exec(layout_optimization, args_mat, process_size, desc='多进程')
        
        # t7 = time.time()
        # print('multi_process_exec: ' + str(t7-t61) + 's.')

        # t8 = time.time()

        # 并行计算结果拆分及目标函数计算
        def sort_key(elem): # 设置一个函数，用于根据布局优化输出中的第二列最短运输距离排序, layout_optimization的输出是 MP_n, DS_best # MP_n 为各台设备的工位编号，其形式为[p1, p2, ……]，按照每个型号设备以及数量，依次对应
            return elem[1]
        
        objects_calculation = []
        SE_set = []
        for i in range(len(i_j)):
            A = []
            B = []
            for j in range(SE_num[i]):
                result = results[0:RE] # 每次取前RE个,对应于每个个体的每种设备数量配置方案下,计算多少次求最佳值
                result.sort(key=sort_key)
                MP_n = result[0][0]
                DS_best = result[0][1]
                CB = CB_calculation(NP, SE_all[i][j], m1_n_all[i], t1_j_all[i], f1_j_all[i], i_j[i], b_nks_all[i], c_nuv_all[i], M, T, F, TJ, TSH, CPM, CAZ, CSV, CSV1) # 系统组建成本计算
                TB = TB_calculation(SE_all[i][j], m1_n_all[i], t1_j_all[i], f1_j_all[i], b_nks_all[i], c_nuv_all[i], M, T, F, TSH, TVD, TVR) # 系统组建时间计算
                TTP = TTP_calculation(NP, SE_all[i][j], m1_n_all[i], m2_j_all[i], t1_j_all[i], f1_j_all[i], i_j[i], a_nr_all[i], b_nks_all[i], J, M, T, F, TJ, AD, TSH1, DS_best, TV) # 产品生产时间计算
                CTP = CTP_calculation(NP, SE_all[i][j], m1_n_all[i], m2_j_all[i], t1_j_all[i], f1_j_all[i], i_j[i], a_nr_all[i], b_nks_all[i], J, M, T, F, TJ, CJ, AD, TSH1, CPM, CZJ, TTP, CD, DS_best, CPR) # 产品生产成本计算
                A.append([CB, TB, CTP, TTP])
                B.append([SE_all[i][j], MP_n])
                del(results[0:RE]) # 当前这RE个结果使用完后, 就删除, 以便于下一个结果直接取前边RE位

            objects_calculation.append(A)
            SE_set.append(B)
        
        # t9 = time.time()
        # print('object calculation: ' + str(t9-t8) + 's.')

        return objects_calculation, SE_set # objects_calculation 一个种群的目标函数值的集合, 有三个维度(个体数量, 设备数量配置方案, 目标数量); SE_set 一个种群的所有个体的设备数量配置与布局方案(每个型号设备选择多少台设备及其位置)，其维度为[个体*设备数量配置与布局方案*[每个设备型号选择的设备数量, 每个设备的位置]]


    '''NSGA-Ⅱ算法寻优'''
    parents = initials
    i_j_P = i_j
    objects_calculation_P, SE_set_P = objects_calculate(i_j_P)

    # t4 = time.time()
    # print('Initialization is finished in ' + str(t4 - t3) + 's.')
    
    print('Configuration optimizing by NSGA-II ......')
    k1 = 0
    k2 = 0
    
    progress_bar(-1, NG1)
    
    fronts0, _, A_best = ENS(N, parents, SE_set_P, i_j_P, objects_calculation_P) # 初始解也做个非支配排序，用来保存初始解的最佳前沿，便于后续结果分析
    objects = [[0, 0] + B[-1] for B in fronts0[0]]
    
    front_best = fronts0[0] # 全局最优前沿
    t40 = time.time()
    while k1 < NG1 and k2 <= NG2:
        childs, i_j_C = crossover_mutation(N, parents, pm, Q_class, i_j_P) # 交叉与变异操作
        P_C = parents + childs
        objects_calculation_C, SE_set_C = objects_calculate(i_j_C) # 计算目标函数值
        SE_set = SE_set_P + SE_set_C
        i_j = i_j_P + i_j_C
        objects_calculation = objects_calculation_P + objects_calculation_C
        fronts, num, A = ENS(N, P_C, SE_set, i_j, objects_calculation) # 非支配排序；fronts 前几个前沿的集合，其中每个二级元素包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]
        S1 = []
        for L in fronts:
            S1 += L
        if num != N:
            S2 = copy.deepcopy(S1)
            A = crowding_distance(S2, fronts, N) # 拥挤距离计算及个体筛选
        k1 += 1
        parents, i_j_P, objects_calculation_P, SE_set_P = update(S1, A)

        t401 = time.time()
        
        front_best, A_best_new = ENS_best(front_best + fronts[0]) # 更新全局最佳前沿
        if A_best == A_best_new: # 如果最优种群长时间未变化，则停止迭代
            k2 += 1
        else:
            k2 = 0
            A_best = A_best_new

        objects += [[k1, t401 - t40] + B[-1] for B in front_best] # 保存每次迭代的最佳前沿，每行形式为[迭代次数，时间，优化目标1，优化目标2，优化目标3，优化目标4]

        # # 记录迭代400次或提前结束时的全局最佳前沿
        # if k1 == 400 or k2 >= NG2: # 导出成csv文件
        #     t41 = time.time()
        #     with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'time_NSGAII_400_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        #         writer = csv.writer(f)
        #         writer.writerow([t41-t1])
        #     with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NSGAII_400_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        #         writer = csv.writer(f)
        #         writer.writerows(front_best)

        progress_bar(k1-1, NG1)

    t5 = time.time()
    # print('Configuration optimizing by NSGA-III is finished in ' + str(t5 - t4) + 's.')
    # print(result)

    # 记录迭代600次或提前结束时的最佳前沿
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NSGAII_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(front_best)
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'time_NSGAII_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerow([t5-t1])
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NSGAII_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(objects)

    # 保存最后一次的种群结果，便于接着迭代计算
    last_iter = []
    for i in range(len(objects_calculation_P)):
        for j in range(len(objects_calculation_P[i])):
            last_iter.append([parents[i], i_j_P[i], SE_set_P[i][j], objects_calculation_P[i][j]]) # 每行[[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]

    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'last_NSGAII_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(last_iter)

    print('Configuration optimizing by NGSA-II is finished.')


# 在终端输入 python main_test_NSGAII.py 运行本程序
