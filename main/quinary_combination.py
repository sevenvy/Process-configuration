import os
import csv
import numpy as np


'''
根据数据间的关联关系生成可行的五元组合（进行了大幅简化）：
特征-工艺：工艺类型
工艺-设备：工艺类型
设备-刀具：刀具接口
工艺-刀具：工艺类型
设备-夹具：夹具接口
'''

'''
# 五元数据的关键参数：
1）特征：特征类型，工艺类型
2）工艺：工艺类型
3）设备：工艺类型，刀具接口，夹具接口，设备价格CM，设备安装调试成本CMD，设备安装调试时间TMA，设备待机单位时间成本CPW，设备平均换刀成本CAH，设备平均换刀时间TAH，设备是否具备多轴加工能力MA，设备交货期TMD，可供应数量NML
4）刀具：工艺类型，刀具接口，刀具价格CST，配件价格CST_1，刀具磨损更换加工时长TST，刀具交货期TTD
5）夹具：夹具接口，夹具价格CSF，夹具安装时间TF，工件装夹时间TF_1，夹具交货期TFD
'''

'''
# 刀轴方向编码：
1-(-1, 0, 0), 2-(1, 0, 0), 3-(0, -1, 0), 4-(0, 1, 0), 5-(0, 0, -1), 6-(0, 0, 1)
'''

if __name__=="__main__": # 如果是其他文件调用这个文件的函数，则只有其他部分会执行，这之后的部分不执行
    # 数据存储路径
    quinary_dir = 'quinary_combination'

    # 数据读取路径
    data_dir = '../database'
    feature_dir = 'feature'
    process_dir = 'process'
    machine_dir = 'machine'
    tool_dir = 'tool'
    fixture_dir = 'fixture'


'''读取数据集'''
# 定义读取数据集的函数
def data_reading(file_path):  # 此处file_path必须为csv文件
    data_txt = []
    with open(file_path, 'r') as f: # 读取csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        reader = csv.reader(f)
        for line in reader:
            data_txt.append(line[1:])  # 去掉第一列的编号，直接用集合中的序号代替编号的作用
    data_txt = data_txt[1:]  # 去掉第一行的列名

    data = []
    for flag1 in range(len(data_txt)):
        data1 = []
        for flag2 in range(len(data_txt[flag1])):
            if data_txt[flag1][flag2][0] == '[':  # 对于包含‘[]’的项，需要将里边的数字提取出来
                a = data_txt[flag1][flag2][1:len(data_txt[flag1][flag2])-1]  # 去掉前后的‘[’和‘]’
                b = a.split(', ') # 对字符串a进行切分
                c = list(map(eval,b)) # 将切分好的字符串序列转换成数字
            elif data_txt[flag1][flag2] == 'max': # 对于刀具集中存在的最大值字符，仍然保留文本
                c = 'max'
            else:
                c = (eval(data_txt[flag1][flag2]))
            data1.append(c)
        data.append(data1)
    return data


if __name__=="__main__":  # 如果是其他文件调用这个文件的函数，则只有其他部分会执行，这之后的部分不执行
    # 特征集
    J = data_reading(os.path.join(data_dir, feature_dir + '.csv'))
    # 工艺集
    P = data_reading(os.path.join(data_dir, process_dir + '.csv'))
    # 设备集
    M = data_reading(os.path.join(data_dir, machine_dir + '.csv'))
    # 刀具集
    T = data_reading(os.path.join(data_dir, tool_dir + '.csv'))


    '''五元数据间两两匹配关系'''
    # 特征-工艺：工艺类型
    J_P = []
    for j1 in range(len(J)):
        J1_P = []
        for p in range(len(P)):
            for j2 in range(len(J[j1][1])):
                if J[j1][1][j2] == P[p][0]:
                    d = 1
                    break
                else:
                    d = 0
            J1_P.append(d)
        J_P.append(J1_P)

    # 工艺-设备：工艺类型
    P_M = []
    for p in range(len(P)):
        P1_M = []
        for m1 in range(len(M)):
            for m2 in range(len(M[m1][0])):
                if M[m1][0][m2] == P[p][0]:
                    #print(M[m1][0])
                    d = 1
                    break
                else:
                    d = 0
            P1_M.append(d)
        P_M.append(P1_M)

    # 设备-刀具：刀具接口
    M_T = []
    for m1 in range(len(M)):
        M1_T = []
        for t in range(len(T)):
            for m2 in range(len(M[m1][1])):
                if M[m1][1][m2] == T[t][1]:
                    d = 1
                    break
                else:
                    d = 0
            M1_T.append(d)
        M_T.append(M1_T)
    print(M_T)

    # 工艺-刀具：工艺类型
    P_T = []
    for p in range(len(P)):
        P1_T = []
        for t in range(len(T)):
            if P[p][0] == T[t][0]:
                d = 1
            else:
                d = 0
            P1_T.append(d)
        P_T.append(P1_T)
    print(P_T)

    '''五元组合的解空间'''
    Q = []
    for j in range(len(J)):
        for p in range(len(P)):
            if J_P[j][p] == 1:
                for m in range(len(M)):
                    if P_M[p][m] == 1:
                        for t in range(len(T)):
                            if M_T[m][t] == 1 and P_T[p][t] == 1:
                                Q.append([j+1, p+1, m+1, t+1])

    # 统计各特征类型下的机床和刀具组合
    face_MT = []
    hole_MT = []
    thread_MT = []
    step_MT = []
    print_MT = []
    for line in Q:
        if J[line[0]-1][0] == 1:  # 基准面
            if [line[2], line[3]] not in face_MT:
                face_MT.append([line[2], line[3]])
                continue
        elif J[line[0]-1][0] == 2 or J[line[0]-1][0] == 3: # 盲孔和通孔
            if [line[2], line[3]] not in hole_MT:
                hole_MT.append([line[2], line[3]])
                continue
        elif J[line[0]-1][0] == 4: # 内螺纹
            if [line[2], line[3]] not in thread_MT:
                thread_MT.append([line[2], line[3]])
                continue
        elif J[line[0]-1][0] == 5 or J[line[0]-1][0] == 6: # 内部台阶和连续内部台阶
            if [line[2], line[3]] not in step_MT:
                step_MT.append([line[2], line[3]])
                continue
        elif J[line[0]-1][0] == 7: # 3D打印
            if [line[2], line[3]] not in print_MT:
                print_MT.append([line[2], line[3]])
                continue
        else:
            print('Error with the quinary combination data, position 1.')
            exit(0)

    # 基准面特征类型下各机床和刀具组合的可能情况
    np.random.seed(25) # 固定随机种子，便于重复随机结果
    face_MT_random = np.random.uniform(low=-1, high=1, size=len(face_MT))

    # 盲孔和通孔特征类型下各机床和刀具组合的可能情况
    np.random.seed(26) # 固定随机种子，便于重复随机结果
    hole_MT_random = np.random.uniform(low=-1, high=1, size=len(hole_MT))

    # 内螺纹特征类型下各机床和刀具组合的可能情况
    np.random.seed(27) # 固定随机种子，便于重复随机结果
    thread_MT_random = np.random.uniform(low=-1, high=1, size=len(thread_MT))

    # 内部台阶和连续内部台阶特征类型下各机床和刀具组合的可能情况
    np.random.seed(28) # 固定随机种子，便于重复随机结果
    step_MT_random = np.random.uniform(low=-1, high=1, size=len(step_MT))

    # 3D打印特征类型下各机床和刀具组合的可能情况
    np.random.seed(29) # 固定随机种子，便于重复随机结果
    print_MT_random = np.random.uniform(low=-1, high=1, size=len(print_MT))

    # 各机床单位时间加工成本随机浮动的可能情况
    np.random.seed(30) # 固定随机种子，便于重复随机结果
    CJ_M = np.random.uniform(low=-1, high=1, size=len(M))

    # 五元组合的加工时间TJ
    i = 0
    for line in Q:
        if J[line[0]-1][2] == 1 or J[line[0]-1][2] == 2 or J[line[0]-1][2] == 3 or J[line[0]-1][2] == 4: # 基准面1-4
            TJ1 = (500 - (M[line[2]-1][2] - 100000) / 900000 * 470) / 3 + 2 * (500 - (T[line[3]-1][2] - 1000) / 6500 * 470) / 3 # （30，500）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：2 + 三分之一上下范围内随机浮动
            TJ = (TJ1 + TJ1 / 3 * face_MT_random[face_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 5 or J[line[0]-1][2] == 6: # 基准面5-6
            TJ1 = (500 - (M[line[2]-1][2] - 100000) / 900000 * 470) / 3 + 2 * (500 - (T[line[3]-1][2] - 1000) / 6500 * 470) / 3 # 同样的设备和刀具情况下为基准1-4的三分之二
            TJ = (TJ1 + TJ1 / 3 * face_MT_random[face_MT.index([line[2], line[3]])]) / 3600 * 2 / 3 
        elif J[line[0]-1][2] == 7: # 盲孔1-4
            if T[line[3]-1][0] == 1: # 铣削
                TJ1 = (100 - (M[line[2]-1][2] - 100000) / 900000 * 75) / 2 + (100 - (T[line[3]-1][2] - 1000) / 6500 * 75) / 2 #  (25：100）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            elif T[line[3]-1][0] == 2: # 钻削
                TJ1 = (24 - (M[line[2]-1][2] - 100000) / 900000 * 8) / 2 + (24 - (T[line[3]-1][2] - 1000) / 6500 * 8) / 2 # （16：24）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            else:
                print('Error with the quinary combination data, position 2.')
                exit(0)            
            TJ = (TJ1 + TJ1 / 4 * hole_MT_random[hole_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 8: # 通孔1-4
            if T[line[3]-1][0] == 1: # 铣削
                TJ1 = (600 - (M[line[2]-1][2] - 100000) / 900000 * 450) / 2 + (600 - (T[line[3]-1][2] - 1000) / 6500 * 450) / 2 #  （150：600）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            elif T[line[3]-1][0] == 2: # 钻削
                TJ1 = (240 - (M[line[2]-1][2] - 100000) / 900000 * 80) / 2 + (240 - (T[line[3]-1][2] - 1000) / 6500 * 80) / 2 #  （160：240）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            else:
                print('Error with the quinary combination data, position 3.')
                exit(0) 
            TJ = (TJ1 + TJ1 / 4 * hole_MT_random[hole_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 9: # 盲孔5-8
            TJ  = (15 + 2.5 * hole_MT_random[hole_MT.index([line[2], line[3]])]) / 3600 #  15秒+上下六分之一范围内随机浮动
        elif J[line[0]-1][2] == 10: # 盲孔9-12
            if T[line[3]-1][0] == 1: # 铣削
                TJ1  = (300 - (M[line[2]-1][2] - 100000) / 900000 * 225) / 2 + (300 - (T[line[3]-1][2] - 1000) / 6500 * 225) / 2 #  （75：300）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            elif T[line[3]-1][0] == 2: # 钻削
                TJ1  = (120 - (M[line[2]-1][2] - 100000) / 900000 * 40) / 2 + (120 - (T[line[3]-1][2] - 1000) / 6500 * 40) / 2 #  （80：120）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            else:
                print('Error with the quinary combination data, position 4.')
                exit(0) 
            TJ = (TJ1 + TJ1 / 4 * hole_MT_random[hole_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 11: # 通孔5
            if T[line[3]-1][0] == 1: # 铣削
                TJ1  = (900 - (M[line[2]-1][2] - 100000) / 900000 * 675) / 2 + (900 - (T[line[3]-1][2] - 1000) / 6500 * 675) / 2 #  （225：900）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            elif T[line[3]-1][0] == 2: # 钻削
                TJ1  = (360 - (M[line[2]-1][2] - 100000) / 900000 * 120) / 2 + (360 - (T[line[3]-1][2] - 1000) / 6500 * 120) / 2 #  （240：360）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+四分之一上下范围内随机浮动
            else:
                print('Error with the quinary combination data, position 5.')
                exit(0) 
            TJ = (TJ1 + TJ1 / 4 * hole_MT_random[hole_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 12 or J[line[0]-1][2] == 13: # 内螺纹1-2
            TJ1 = (40 - (M[line[2]-1][2] - 100000) / 900000 * 20) / 2 + (40 - (T[line[3]-1][2] - 1000) / 6500 * 20) / 2 # （20：40）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+六分之一上下范围内随机浮动
            TJ = (TJ1 + TJ1 / 6 * thread_MT_random[thread_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 14 or J[line[0]-1][2] == 16 or J[line[0]-1][2] == 18: # 内部台阶1，3，5
            TJ1 = (90 - (M[line[2]-1][2] - 100000) / 900000 * 45) / 2 + (90 - (T[line[3]-1][2] - 1000) / 6500 * 45) / 2 # （45，90）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+六分之一上下范围内随机浮动
            TJ = (TJ1 + TJ1 / 6 * step_MT_random[step_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 15 or J[line[0]-1][2] == 17 or J[line[0]-1][2] == 19: # 内部台阶2，4，连续内部台阶1
            TJ1 = (180 - (M[line[2]-1][2] - 100000) / 900000 * 90) / 2 + (180 - (T[line[3]-1][2] - 1000) / 6500 * 90) / 2 # （90，180）秒范围内与机床和刀具价格成正比，机床和刀具价格的权重为1：1+六分之一上下范围内随机浮动
            TJ = (TJ1 + TJ1 / 6 * step_MT_random[step_MT.index([line[2], line[3]])]) / 3600
        elif J[line[0]-1][2] == 20: # 3D打印体1
            TJ1 = 0.65 - (M[line[2]-1][2] - 100000) / 900000 * 0.525 # （0.125，0.65）小时范围内与机床价格成正比+四分之一上下范围内随机浮动
            TJ = TJ1 + TJ1 / 4 * print_MT_random[print_MT.index([line[2], line[3]])]
        else:
            print('Error with the quinary combination data, position 6.')
            exit(0)
        
        Q[i].append(TJ) # 每行添加一个参数TJ
        i += 1

    # 五元组合的加工成本CJ
    for i in range(len(Q)):
        if T[Q[i][3]-1][0] != 7: # 非3D打印：（待机成本*4+上下四分之一范围内浮动）*加工时间
            CJ = (M[Q[i][2]-1][3] * 4 + M[Q[i][2]-1][3] * CJ_M[Q[i][2]-1]) * Q[i][4]
        elif T[Q[i][3]-1][0] == 7:
            if M[Q[i][2]-1][0] == [7]: # 纯3D打印设备的3D打印：（待机成本*8+上下四分之一范围内浮动）*加工时间
                CJ = (M[Q[i][2]-1][3] * 8 + 2 * M[Q[i][2]-1][3] * CJ_M[Q[i][2]-1]) * Q[i][4]
            else: # 混合加工设备的3D打印：（待机成本*6+上下四分之一范围内浮动）*加工时间
                CJ = (M[Q[i][2]-1][3] * 6 + 1.5 * M[Q[i][2]-1][3] * CJ_M[Q[i][2]-1]) * Q[i][4]
        else:
            print('Error with the quinary combination data, position 7.')
            exit(0)        

        Q[i].append(CJ) # 每行添加一个参数CJ

    # 给五元组合前边添加一列编号
    for i in range(len(Q)):
        Q[i] = [i] + Q[i]

    # 导出成csv文件
    with open(os.path.join('../database', quinary_dir + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerow(['quinary combination index', 'J Feature', 'P Process', 'M Machine', 'T Tool', 'TJ Processing duration', 'CJ Processing cost'])  # 给每一列加上列名
        writer.writerows(Q)

    print('Quinary combination set generation is finished.')

    # 在终端输入 python quinary_combination.py 运行本程序