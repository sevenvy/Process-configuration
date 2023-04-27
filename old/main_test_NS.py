import os
import math
import time
import random
import csv


'''本程序仅在数值实验时使用'''

# 从其他python文件中调用函数
from quinary_combination import data_reading
from machine_number import NM_calculation
from layout_optimization import DMP_calculation, layout_optimization
from optimization_objects import variable_determine, CB_calculation, TB_calculation, CTP_calculation, TTP_calculation
from NSGAIII_algorithm import generate_initials
from MOSA_algorithm import ENS_4
from multi_process import multi_process_exec
from NS_algorithm import neighbour_P, neighbour_M, neighbour_T, neighbour_F, ENS_1, neighbour_new, ENS_6


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

    # 禁忌搜索算法参数设置
    T_size = 15 # 禁忌表的大小
    S_size = 20 # 候选集合的大小
    TG1 = 500 # 总体最大迭代次数 15000
    TG2 = 200 # 历史最优解持续未变化的最大迭代次数 5000
    RE = 1 # 重复迭代多少次求最小值

    # 计算线程/进程参数设置
    process_size = 60 # 布局优化进程数 # 不能超过60
    # process_size2 = 15 # 不同设备配置情况下的布局优化进程数

    # NS算法参数设置
    max_num = 50 # 每次迭代邻域解集抽取的解的个数
    SG1 = 600 # 总体最大迭代次数
    SG2 = 200 # 最优解集持续未变化的最大迭代次数


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

    
    '''定义一个函数用于计算目标函数值'''
    def objects_calculate(i_j):

        # t6 = time.time()

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
    initials, i_j = generate_initials(Q_class, max_num)

    
    # # 读取前置算法得到的最佳前沿数据
    # data_txt = []
    # with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NSGAIII_400_' + str(test_num) + '.csv'), 'r') as f: # 读取csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
    #     reader = csv.reader(f)
    #     for line in reader:
    #         data_txt.append(line) # 去掉第一列的编号，直接用集合中的序号代替编号的作用
    # result = []
    # for flag1 in range(len(data_txt)):
    #     data1 = []
    #     for flag2 in range(len(data_txt[flag1])):
    #         if data_txt[flag1][flag2][0:2] == '[[': # 对于包含‘[[]]’的项，需要将里边的数字提取出来
    #             a = data_txt[flag1][flag2][2:len(data_txt[flag1][flag2])-2] # 去掉最外层的‘[[’和‘]]’
    #             b = a.split('], [') # 对字符串a进行切分
    #             c = []
    #             for b1 in b:
    #                 b2 = b1.split(', ') # 对字符串b1进行切分
    #                 b3 = list(map(eval,b2)) # 将切分好的字符串序列转换成数字
    #                 c.append(b3)
    #         elif data_txt[flag1][flag2][0:1] == '[': # 对于包含‘[]’的项，需要将里边的数字提取出来
    #             a = data_txt[flag1][flag2][1:len(data_txt[flag1][flag2])-1] # 去掉最外层的‘[’和‘]’
    #             b = a.split(', ') # 对字符串a进行切分
    #             c = list(map(eval,b)) # 将切分好的字符串序列转换成数字
    #         data1.append(c)
    #     result.append(data1)

    print('Configuration optimizing by NS ......')
    # t51 = time.time()

    '''NS算法寻优'''
    # # 初始解
    # X = []
    # objects_X = []
    # SE_X = []
    # for a in result: # 统计最佳前沿中所涉及的解
    #     if a[0:2] not in X:
    #         X.append(a[0:2])
    #         objects_X.append([a[-1]])
    #         SE_X.append([a[2]])
    #     else:
    #         b = X.index(a[0:2])
    #         objects_X[b].append(a[-1])
    #         SE_X[b].append(a[2])

    # # 初始解
    # X0 = []
    # objects_X0 = []
    # SE_X0 = []
    # for a in result: # 统计最佳前沿中所涉及的解
    #     if a[0:2] not in X0:
    #         X0.append(a[0:2])
    #         objects_X0.append([a[-1]])
    #         SE_X0.append([a[2]])
    #     else:
    #         b = X0.index(a[0:2])
    #         objects_X0[b].append(a[-1])
    #         SE_X0[b].append(a[2])

    # # 按照最佳前沿中涉及各解的元素数量选取max_num个解
    # obj_len = [] # 用于记录各解的元素数量
    # for i in range(len(objects_X0)):
    #     obj_len += [[i, len(objects_X0[i])]]
    # obj_len.sort(key=lambda x: x[1], reverse=True)
    
    # X = []
    # objects_X = []
    # SE_X = []
    # for i in range(max_num):
    #     X += [X0[obj_len[i][0]]]
    #     objects_X += [objects_X0[obj_len[i][0]]]
    #     SE_X += [SE_X0[obj_len[i][0]]]


    # 初始解
    X = [[initials[i], i_j[i]] for i in range(len(i_j))]
    objects_X, SE_X = objects_calculate(i_j)

    front, As, i_j_A, objects_A, SE_A = ENS_4(initials, i_j, objects_X, SE_X) # 初始解也做个非支配排序，用来保存初始解的最佳前沿，便于后续结果分析
    objects = [[0, 0] + B[-1] for B in front]


    # print(len(result))
    # print(len(X))

    # 搜索过程
    # front = result
    # objects = []
    # k2 = 0
    # neighbour_func = [neighbour_P, neighbour_M, neighbour_T, neighbour_F]
    t52 = time.time()

    progress_bar(-1, SG1)

    # for k1 in range(SG1):
    #     XN = []
    #     for x in X: # 依次生成邻域解集，并分别从中随机抽取max_num个解
    #         for i in range(len(neighbour_func)):
    #             XN += neighbour_func[i](x, Q_class) # 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
    #     if len(XN) >= max_num:
    #         X1 = random.sample(XN, max_num)
    #     else:
    #         X1 = XN
    #     i_j_1 = [a[1] for a in X1]
    #     objects_calculation_1, SE_set_1 = objects_calculate(i_j_1)

    #     # 非支配排序
    #     F_N = X + X1
    #     objects_calculation = objects_X + objects_calculation_1
    #     SE_set = SE_X + SE_set_1
    #     front, XF, objects_XF, SE_XF = ENS_1(F_N, objects_calculation, SE_set) # front为最佳前沿，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]
        
    #     if XF != X:
    #         k2 = 0
    #     else:
    #         k2 += 1
    #     if k2 >= SG2:
    #         break
            
    #     X = XF
    #     objects_X = objects_XF
    #     SE_X = SE_XF

    #     t53 = time.time()
    #     objects += [[k1, t53 - t52] + C[-1] for C in front] # 保存每次迭代的最佳前沿，每行形式为[迭代次数，时间，优化目标1，优化目标2，优化目标3，优化目标4]


    Ta = [[] for _ in range(len(X))] # 记录各解的已选邻域解，避免重复计算
    k2 = [0 for _ in range(len(X))] # 记录各解保持不变的迭代次数
    # F_best = result.copy() # 全局最优前沿
    F_best = front.copy() # 全局最优前沿
    for k1 in range(SG1):
        X_new = []
        flag = 0
        for i1 in range(len(X)): # 依次生成邻域解
            i = i1 - flag
            x_N = neighbour_new(X[i], Q_class) # 邻域解集，其每个元素的形式为[[各加工特征的五元组合], [各五元组合编号]]
            x_N1 = list(filter(lambda r: r not in Ta[i], x_N)) # 筛选出尚未选择的解
            if x_N1 != []:
                X_new.append(random.sample(x_N1,1)[0])
            else:
                del(X[i])
                del(objects_X[i])
                del(SE_X[i])
                del(k2[i])
                flag += 1
        
        i_j_1 = [a[1] for a in X_new]
        objects_calculation_1, SE_set_1 = objects_calculate(i_j_1)

        # 非支配排序
        fronts = []
        flag = 0
        for i1 in range(len(X)):
            i = i1 - flag
            F_N = [X[i], X_new[i]]
            objects_calculation = [objects_X[i], objects_calculation_1[i]]
            SE_set = [SE_X[i], SE_set_1[i]]
            front, _, _, _ = ENS_1(F_N, objects_calculation, SE_set) # front为最佳前沿，其中包含 [[各加工特征的五元组合], [各五元组合编号], [设备数量配置与布局方案], [各目标函数值]]
            Xnew_in_F = list(filter(lambda a: a[0:2] in X_new, front))
            xnew_F_num = len(Xnew_in_F)
            x_F_num = len(front) - xnew_F_num
            if x_F_num <= xnew_F_num:
                Ta[i] = [X[i].copy()]
                X[i] = X_new[i]
                objects_X[i] = objects_calculation_1[i]
                SE_X[i] = SE_set_1[i]
                k2[i] = 0
            else:
                Ta[i].append(X_new[i].copy())
                k2[i] += 1
                if k2[i] >= SG2:
                    del(X[i])
                    del(objects_X[i])
                    del(SE_X[i])
                    del(k2[i])
                    flag += 1  

            fronts += Xnew_in_F
        F_best = ENS_6(F_best + fronts) # 更新全局最优前沿

        t53 = time.time()
        objects += [[k1, t53 - t52] + C[-1] for C in F_best] # 保存每次迭代的最佳前沿，每行形式为[迭代次数，时间，优化目标1，优化目标2，优化目标3，优化目标4]
    
        progress_bar(k1, SG1)

    t6 = time.time()
    
    # 记录迭代600次或提前结束时的最佳前沿
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'best_front_NS_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(F_best)
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'time_NS_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        # writer.writerow([t6-t51])
        writer.writerow([t6-t1])
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'object_NS_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(objects)

    # 保存最后一次的种群结果，便于接着迭代计算
    last_iter = X
    with open(os.path.join('results', 'feature_solution_' + str(feature_solution_num), 'last_MOSA_600_' + str(test_num) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerows(last_iter)

    print('Configuration optimizing by NS is finished.')


# 在终端输入 python main_test_NS.py 运行本程序
