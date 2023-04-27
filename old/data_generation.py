import os
import csv
import numpy as np


'''
# 五元数据的关键参数：
1）特征：特征类型，工艺类型，特征序号
2）工艺：工艺类型
3）设备：工艺类型，刀具接口，夹具接口，设备价格CM，设备安装调试成本CMD，设备安装调试时间TMA，设备待机单位时间成本CPW，设备平均换刀成本CAH，设备平均换刀时间TAH，设备是否具备多轴加工能力MA，设备交货期TMD，可供应数量NML
4）刀具：工艺类型，刀具接口，刀具价格CST，配件价格CST_1，刀具磨损更换加工时长TST，刀具交货期TTD
5）夹具：夹具接口，夹具价格CSF，夹具安装时间TF，工件装夹时间TF_1，夹具交货期TFD
'''

'''
# 加工特征类型编码：
1-基准面，2-盲孔，3-通孔，4-内螺纹，5-内部台阶，6-连续内部台阶，7-3D打印体
# 工艺类型编码：
1-铣削，2-钻削，3-车削，4-攻丝，5-镗削，6-特种镗削，7-3D打印
# 刀具接口类型编码：
1-HSK-A63，2-HSK-E63，3-SK30，4-SK40，5-BT30，6-BT40，7-Walter capto，8-无需安装刀具
# 夹具接口类型编码：
1-T型槽，2-三爪卡盘，3-法兰盘，4-磁吸，5-无需安装夹具
'''

# 数据量
NM = 30 # 设备型号数量 100
NT = 50 # 刀具型号数量 300
NF = 30 # 夹具型号数量 100

# 数据存储路径
data_dir = 'database'
feature_dir = 'feature'
process_dir = 'process'
machine_dir = 'machine'
tool_dir = 'tool'
fixture_dir = 'fixture'


"""加工特征集"""
# 加工特征，共有四种可选方案
feature_solution_num = 1
feature_solutions = []
feature_solutions.append([1,1,1,1,1,1,2,3,2,2,3,4,4,5,5,5,5,5])
feature_solutions.append([1,1,1,1,1,1,2,3,2,2,3,4,4,6])
feature_solutions.append([7,1,1,1,1,1,1,2,2,3,4,4,5,5,5,5,5])
feature_solutions.append([7,1,1,1,1,1,1,2,2,3,4,4,6])
feature_nums = []
feature_nums.append([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
feature_nums.append([1,2,3,4,5,6,7,8,9,10,11,12,13,19])
feature_nums.append([20,1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,18])
feature_nums.append([20,1,2,3,4,5,6,9,10,11,12,13,19])



# 设置一个简单的函数，构建五元数据中加工特征的参数组
def feature_constraction(feature_solution, feature_num):
    J = []
    for j in range(len(feature_solution)):
        if feature_solution[j] == 1:
            J.append([j+1,1,[1],feature_num[j]])
        elif feature_solution[j] == 2:
            J.append([j+1,2,[1,2],feature_num[j]])
        elif feature_solution[j] == 3:
            J.append([j+1,3,[1,2],feature_num[j]])
        elif feature_solution[j] == 4:
            J.append([j+1,4,[4],feature_num[j]])
        elif feature_solution[j] == 5:
            J.append([j+1,5,[5],feature_num[j]])
        elif feature_solution[j] == 6:
            J.append([j+1,6,[6],feature_num[j]])
        elif feature_solution[j] == 7:
            J.append([j+1,7,[7],feature_num[j]])
        else:
            print('Error with the feature data.')
            exit(0)
    return J

# 导出成csv文件
for num in range(feature_solution_num): 
    J = feature_constraction(feature_solutions[num], feature_nums[num])    
    with open(os.path.join(data_dir, feature_dir + '_' + str(num + 1) + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
        writer = csv.writer(f)
        writer.writerow(['Feature index','Feature type','Process type','Feature number']) # 给每一列加上列名：特征编号，特征类型，工艺类型
        writer.writerows(J)


"""工艺集"""
P = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]

# 导出成csv文件
with open(os.path.join(data_dir, process_dir + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
    writer = csv.writer(f)
    writer.writerow(['Process index','Process type']) # 给每一列加上列名：特征编号，特征类型，工艺类型
    writer.writerows(P)


"""设备集"""
# 设备集的生成方式是，以现有设备为基础，给出限定范围，随机生成

# 工艺类型的可能情况
np.random.seed(1) # 固定随机种子，便于重复随机结果
M_1 = np.random.randint(low=1, high=8, size=NM) # 设备工艺类型共有7种可能，随机生成

# 刀具接口类型的可能情况
# 设备上有一种刀具接口
np.random.seed(2) # 固定随机种子，便于重复随机结果
M_2_1 = np.random.randint(low=1, high=8, size=NM) # 刀具接口类型共有7种可能（除无需安装刀具外），随机生成
# 有30%的可能存在两种刀具接口
np.random.seed(3) # 固定随机种子，便于重复随机结果
a = np.random.random(size=NM)
b = []
for m in range(NM):
    if a[m] < 0.7:
        b.append(0)
    else:
        b.append(1)
np.random.seed(4) # 固定随机种子，便于重复随机结果
M_2_2 = np.random.randint(low=1, high=8, size=NM) # 随机生成
M_2 = []
for m in range(NM):
    if b[m] == 0:
        M_2.append([M_2_1[m]])
    else:
        M_2.append([M_2_1[m],M_2_2[m]])

# 设备是否具备多轴加工能力的可能情况
# 随机生成，10%的可能性具备多轴加工能力
np.random.seed(5) # 固定随机种子，便于重复随机结果
c = np.random.random(size=NM)
M_10 = []
for m in range(NM):
    if c[m] < 0.9:
        M_10.append(0)
    else:
        M_10.append(1)

# 设备价格CM上下浮动随机因子的可能情况
np.random.seed(6) # 固定随机种子，便于重复随机结果
CM_2 = np.random.uniform(low=(-1), high=1, size=NM)

# 设备安装调试成本CMD上下浮动随机因子的可能情况
np.random.seed(7) # 固定随机种子，便于重复随机结果
CMD_2 = np.random.uniform(low=(-1), high=1, size=NM)

# 设备安装调试时间TMA上下浮动随机因子的可能情况
np.random.seed(8) # 固定随机种子，便于重复随机结果
TMA_2 = np.random.uniform(low=(-1), high=1, size=NM)

# 设备待机单位时间成本CPW上下浮动随机因子的可能情况
np.random.seed(9) # 固定随机种子，便于重复随机结果
CPW_2 = np.random.uniform(low=0.5, high=1, size=NM)

# 设备平均换刀成本CAH上下浮动随机因子的可能情况
np.random.seed(10) # 固定随机种子，便于重复随机结果
CAH_2 = np.random.uniform(low=0.1, high=0.5, size=NM)

# 设备平均换刀时间TAH上下浮动随机因子的可能情况
np.random.seed(11) # 固定随机种子，便于重复随机结果
TAH_2 = np.random.uniform(low=0.002, high=0.01, size=NM)

# 设备交货期TMD上下浮动随机因子的可能情况
np.random.seed(12) # 固定随机种子，便于重复随机结果
TMD = np.random.randint(low=72, high=721, size=NM)

# 设备可供应数量NML上下浮动随机因子的可能情况
np.random.seed(13) # 固定随机种子，便于重复随机结果
NML = np.random.randint(low=1, high=16, size=NM)


# 将参数整合
M = []
for m in range(NM):
    machine = [m+1] # 最为第一列的设备型号编号
    
    # 每个型号设备的工艺类型
    if M_1[m] == 1:
        machine.append([1,2,4]) # 设备工艺类型1：铣，钻，攻
    elif M_1[m] == 2:
        machine.append([1,2,4,5]) # 设备工艺类型2：铣，钻，攻，镗
    elif M_1[m] == 3:
        machine.append([2,4]) # 设备工艺类型3：钻，攻
    elif M_1[m] == 4:
        machine.append([2,3,5]) # 设备工艺类型4：钻，车，镗
    elif M_1[m] == 5:
        machine.append([1,2,4,5,6]) # 设备工艺类型5：铣，钻，攻，镗，特镗
    elif M_1[m] == 6:
        machine.append([7]) # 设备工艺类型6：3D打印      
    elif M_1[m] == 7:
        machine.append([1,2,4,5,7]) # 设备工艺类型5：铣，钻，攻，镗，3D打印
    else:
        print('Error with the machine data.')
        exit(0)
    
    # 每个型号设备的刀具接口类型
    if M_1[m] == 7:
        machine.append([8])
    else:
        machine.append(M_2[m])
    
    # 每个型号设备的夹具接口类型
    if M_1[m] == 1 or M_1[m] == 2 or M_1[m] == 3 or M_1[m] == 5 or M_1[m] == 7: # 除车床和3D打印外，都是T型槽或磁吸
        machine.append([1,4])
    elif M_1[m] == 4: # 车床是三爪卡盘或法兰盘
        machine.append([2,3])
    elif M_1[m] == 6: # 3D打印无需安装夹具
        machine.append([5])
    else:
        print('Error with the machine data.')
        exit(0)    

    # 每个型号设备的价格CM
    if M_1[m] == 1 or M_1[m] == 4: # 工艺类型的系数分别为2.5，3，2，2.5，4，4，6
        process_coefficient = 2.5
    elif M_1[m] == 2:
        process_coefficient = 3
    elif M_1[m] == 3:
        process_coefficient = 2
    elif M_1[m] == 5 or M_1[m] == 6:
        process_coefficient = 4
    elif M_1[m] == 7:
        process_coefficient = 6
    else:
        print('Error with the machine data.')
        exit(0)          
    
    if M_2[m][0] == 1 or M_2[m][0] == 2 or (len(M_2[m]) == 2 and (M_2[m][1] == 1 or M_2[m][1] == 2)): # 刀具接口的系数分别为1.5，1.5，1，1，1，1，1.2，1
        tool_coefficient = 1.5
    elif M_2[m][0] == 7 or (len(M_2[m]) == 2 and M_2[m][1]) == 7:
        tool_coefficient = 1.2
    else:
        tool_coefficient = 1
    
    if M_10[m] == 0: # 多轴加工能力系数分别为是则为2，否则为1
        multiaxis_coefficient = 1
    else:
        multiaxis_coefficient = 2

    CM_1 = 50000 * process_coefficient * tool_coefficient * multiaxis_coefficient
    CM = CM_1 + CM_1 / 5 * CM_2[m] # 在算出的价格的上下五分之一范围内浮动
    machine.append(CM)

    # 每个型号设备的安装调试成本CMD
    CMD_1 = CM / 100 + CM / 500 * CMD_2[m] # 价格的百分之一+上下随机浮动（上下五分之一范围）
    if CMD_1 < 1000: # 如果小于1000则加500
        CMD = CMD_1 + 500
    else:
        CMD = CMD_1 
    machine.append(CMD)

    # 每个型号设备的安装调试时间TMA
    TMA_1 = 6 + 6 * len(machine[1]) * multiaxis_coefficient # 设备搬运，供电、气等6小时+6小时*工艺类型数*多轴加工能力系数
    TMA = TMA_1 + TMA_1 / 3 * TMA_2[m] # 上下随机浮动（时间的三分之一范围内浮动）
    machine.append(TMA)

    # 每个型号设备的待机单位时间成本CPW
    CPW = (5 + len(machine[1])) * CPW_2[m] # （5+工艺类型数）*（0.5，1）范围内的随机数
    machine.append(CPW)

    # 每个型号设备的平均换刀成本CAH
    if M_1[m] == 6: # 如果无需安装刀具则为0
        machine.append(0)
    else:
        machine.append(CAH_2[m]) # （0.1，0.5）范围内随机

    # 每个型号设备的平均换刀时间TAH
    if M_1[m] == 6: # 如果无需安装刀具则为0
        machine.append(0)
    else:
        machine.append(TAH_2[m]) # （0.002，0.01）范围内随机

    # 每个型号设备是否具备多轴加工能力MA
    machine.append(M_10[m])

    # 每个型号设备的交货期
    machine.append(TMD[m]) # 3天-30天范围内随机

    # 每个型号设备的可供应数量NML
    machine.append(NML[m])

    M.append(machine)

# 导出成csv文件
with open(os.path.join(data_dir, machine_dir + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
    writer = csv.writer(f)
    writer.writerow(['Machine index','Process type', 'Tool interface', 'Fixture interface', 'CM Machine price', 'CMD Installation and commissioning cost', 'TMA Installation and commissioning duration',\
    'CPW Unit time cost of waiting', 'CAH Average tool change cost', 'TAH Average tool change duration', 'MA Multi axis processing capability', 'TMD Date of delivery', 'NML Quantity available']) # 给每一列加上列名
    writer.writerows(M)


"""刀具集"""
# 工艺类型的可能情况
np.random.seed(14) # 固定随机种子，便于重复随机结果
T_1 = np.random.randint(low=1, high=8, size=NT-1) # 刀具工艺类型共有7种可能，随机生成,留着最后一个刀具型号给无需安装刀具的纯3D打印设备

# 刀具接口类型的可能情况
np.random.seed(15) # 固定随机种子，便于重复随机结果
T_2 = np.random.randint(low=1, high=8, size=NT-1) # 刀具接口类型共有7种可能，随机生成,留着最后一个刀具型号给无需安装刀具的纯3D打印设备

# 刀具价格CST的随机浮动的可能情况
np.random.seed(16) # 固定随机种子，便于重复随机结果
CST_2 = np.random.uniform(low=(-1), high=1, size=NT-1) # 留着最后一个刀具型号给无需安装刀具的纯3D打印设备

# 配件价格CST1的随机浮动的可能情况
np.random.seed(17) # 固定随机种子，便于重复随机结果
CST1_2 = np.random.uniform(low=(-1), high=1, size=NT-1) # 留着最后一个刀具型号给无需安装刀具的纯3D打印设备

# 刀具磨损更换加工时长TST的随机浮动的可能情况
np.random.seed(18) # 固定随机种子，便于重复随机结果
TST_2 = np.random.uniform(low=0, high=1, size=NT-1) # 留着最后一个刀具型号给无需安装刀具的纯3D打印设备

# 刀具交货期TTD的随机浮动的可能情况
np.random.seed(19) # 固定随机种子，便于重复随机结果
TTD = np.random.randint(low=72, high=481, size=NT-1) # 留着最后一个刀具型号给无需安装刀具的纯3D打印设备


# 将参数整合
T = []
for t in range(NT-1):
    tool = [t+1] # 刀具库中第一列为编号
    
    # 每个型号刀具的工艺类型
    tool.append(T_1[t])

    # 每个型号刀具的刀具接口类型
    tool.append(T_2[t])

    # 每个型号刀具的价格CST
    if T_1[t] == 1 or T_1[t] == 2 or T_1[t] == 4: # 工艺类型的系数分别为1，1，1.2，1，1.2，2，5
        process_coefficient = 1
    elif T_1[t] == 3 or T_1[t] == 5:
        process_coefficient = 1.2
    elif T_1[t] == 6:
        process_coefficient = 2
    elif T_1[t] == 7:
        process_coefficient = 5
    else:
        print('Error with the tool data.')
        exit(0)              
    
    if T_2[t] == 1 or T_2[t] == 2 or T_2[t] == 7: # 刀具接口类型的系数分别为1.5，1.5，1.2，1.2，1，1，1.5
        tool_coefficient = 1.5
    elif T_2[t] == 3 or T_2[t] == 4:
        tool_coefficient = 1.2
    elif T_2[t] == 5 or T_2[t] == 6:
        tool_coefficient = 1
    else:
        print('Error with the tool data.')
        exit(0)       

    CST_1 = 1000 * process_coefficient * tool_coefficient # 1000*工艺类型系数*刀具接口类型系数
    CST = CST_1 + CST_1 / 4 * CST_2[t] # 上下四分之一范围随机浮动
    tool.append(CST)

    # 每个型号刀具的配件价格CST1
    if T_1[t] == 1 or T_1[t] == 2: # 工艺类型的系数分别为1.5，1.5，1，3，3，6，0，其中0表示3D打印头无需更换配件
        process_coefficient = 1.5
    elif T_1[t] == 3:
        process_coefficient = 1
    elif T_1[t] == 4 or T_1[t] == 5:
        process_coefficient = 3
    elif T_1[t] == 6:
        process_coefficient = 6
    elif T_1[t] == 7:
        process_coefficient = 0
    else:
        print('Error with the tool data.')
        exit(0)      

    CST1_1 = 50 * process_coefficient # 50*工艺类型系数
    CST1 = CST1_1 + CST1_1 / 2 * CST1_2[t] # 上下二分之一范围随机浮动
    tool.append(CST1)

    # 每个型号刀具的磨损更换加工时长TST
    if T_1[t] == 1:
        tool.append(20 + 30 * TST_2[t]) # 铣削，（20，50）范围内随机
    elif T_1[t] == 2:
        tool.append(0.3 + 0.7 * TST_2[t]) # 钻削，（0.3，1）范围内随机
    elif T_1[t] == 3 or T_1[t] == 5 or T_1[t] == 6:
        tool.append(1 + 3 * TST_2[t]) # 车削，（1，4）范围内随机；镗削，（1，4）范围内随机；特种镗削，（1，4）范围内随机
    elif T_1[t] == 4:
        tool.append(1 + 2 * TST_2[t]) # 攻丝，（1，3）范围内随机
    elif T_1[t] == 7:
        tool.append('max') # 3D打印，max
    else:
        print('Error with the tool data.')
        exit(0)    
    
    # 每个型号刀具的交货期TTD
    tool.append(TTD[t])

    T.append(tool)

T.append([NT, 7, 8, 0, 0, 'max', 0]) # 最后一个空刀具型号给无需安装刀具的纯3D打印设备

# 导出成csv文件
with open(os.path.join(data_dir, tool_dir + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
    writer = csv.writer(f)
    writer.writerow(['Tool index','Process type', 'Tool interface', 'CST Tool price', 'CST1 Accessory price', 'TST Processing duration to tool wear & replacement','TTD Date of delivery']) # 给每一列加上列名
    writer.writerows(T)


"""夹具集"""
# 夹具接口类型的可能情况
np.random.seed(20) # 固定随机种子，便于重复随机结果
d = np.random.randint(low=1, high=15, size=NF-1) # 夹具接口类型共有4种可能，按10：2：1：1的比例随机生成（在后边代码中判断）,留着最后一个夹具型号给无需安装夹具的3D打印

# 夹具价格CSF的可能情况
np.random.seed(21) # 固定随机种子，便于重复随机结果
F_2 = np.random.randint(low=500, high=10001, size=NF-1) # （500，10000）范围内随机

# 夹具安装时间TF的可能情况
np.random.seed(22) # 固定随机种子，便于重复随机结果
F_3 = np.random.uniform(low=0.2, high=1, size=NF-1) # （0.2，1）范围内随机

# 工件装夹时间TF1的随机浮动的可能情况
np.random.seed(23) # 固定随机种子，便于重复随机结果
TF1_2 = np.random.uniform(low=(-1), high=1, size=NF-1)

# 夹具交货期TFD的可能情况
np.random.seed(24) # 固定随机种子，便于重复随机结果
TFD = np.random.randint(low=72, high=481, size=NF-1)


# 参数整合
F = []
for f in range(NF-1):
    fixture = [f+1] # 夹具库中第一列为编号

    # 每个型号夹具的夹具接口
    if d[f] <= 10: # 按10：2：1：1的比例随机的
        fixture.append(1)
    elif d[f] == 11 or d[f] == 12:
        fixture.append(2)
    elif d[f] == 13:
        fixture.append(3)
    elif d[f] == 14:
        fixture.append(4)
    else:
        print('Error with the fixture data.')
        exit(0)          

    # 每个型号夹具的价格CSF
    fixture.append(F_2[f])

    # 每个型号夹具的安装时间TF
    fixture.append(F_3[f])

    # 每个型号夹具的工件装夹时间TF1
    TF1_1 = 0.015 - 0.015 / (10000 - 500) * (F_2[f] - 500)
    TF1 = TF1_1 + TF1_1 / 5 * TF1_2[f] # 与夹具价格负相关，0.015-0.015/（10000-500）*（夹具价格-500）+五分之一上下范围内随机波动
    fixture.append(TF1)

    # 每个型号夹具的交货期TFD
    fixture.append(TFD[f])

    F.append(fixture)

F.append([NF, 5, 0, 0, 0, 0]) # 最后一个空夹具型号给无需安装夹具的3D打印

# 导出成csv文件
with open(os.path.join(data_dir, fixture_dir + '.csv'), 'w', newline='') as f: # 保存成csv文件，可参考：https://www.cnblogs.com/xbhog/p/13141128.html
    writer = csv.writer(f)
    writer.writerow(['Fixture index', 'Fixture interface', 'CSF Fixture price', 'TF Installation duration', 'TF1 Workpiece clamping duration','TTF Date of delivery']) # 给每一列加上列名
    writer.writerows(F)

print('Data sets generation is finished.')


