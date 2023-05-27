import math
import pandas as pd

if __name__ == "__main__":  # 如果是其他文件调用这个文件的函数，则只有其他部分会执行，这之后的部分不执行
    NP = 200  # 待加工产品数量
    TSH = 0.2  # 安装刀具时间
    TSH1 = 0.05  # 刀具磨损更换时间
    TV = 1800  # 运输系统平均运输速率
    CD = 0.3  # 单位运输距离的物料运输成本
    CRP = 50  # 原材料成本
    Combination = [[17,1,1,30,37], [100,6,1,30,2], [27,2,1,15,23], [57,4,1,10,12]]


m = pd.read_csv('../database/machine.csv')
t = pd.read_csv('../database/tool.csv')
quinary = pd.read_csv('../database/quinary_combination.csv')
'''适应度函数计算'''
def adapt(combination):
    """
    :param combination: 所需要计算的三元组合
    :param jpmtf,quinary:  设备刀具夹具的数据集，用于计算适应度,数据类型：pd.dataframes
    :return: [设备负载率，加工时间，加工成本]
    """
    # 与设备刀具三元组合均无关的常量
    NP = 200  # 待加工产品数量
    TT = 0.2  # 安装刀具时间
    TT1 = 0.05  # 刀具磨损更换时间
    TV = 1800  # 运输系统平均运输速率
    CD = 0.3  # 单位运输距离的物料运输成本
    CP = 50  # 原材料成本

    # 变量
    machine = []  # 该三元组合的设备集
    rr = []  # 每台设备对应的特征集
    front = 0
    TS = 0  # 第一件产品的生产时间
    NML = 0  # 设备所需最小运行空间
    TMP = []
    TMA = []
    CMD = 0
    CMF = 0
    CMU = 0
    for process in combination:
        TS += quinary.at[process[0], 'TJ Processing duration']
        CMD += NP * quinary.at[process[0], 'CJ Processing cost']
        if process[3] != front:
            machine.append(process[3])
            front = process[3]
    i = 0
    while i < len(combination):
        j = i + 1
        while j < len(combination) and combination[j][3] == combination[i][3]:
            j += 1
        rr.append(list(range(i, j)))
        i = j
    for feature_set in rr:
        tmp_m = 0
        tmh_m = 0
        for each_f in range(len(feature_set)):  # 遍历每个特征
            process_time = quinary.at[combination[feature_set[each_f]][0], 'TJ Processing duration'].astype(float)
            if t.at[combination[feature_set[each_f]][4], 'TST Processing duration to tool wear'] == 'max':
                k_j = 0
            else:
                tst = float(t.at[combination[feature_set[each_f]][4], 'TST Processing duration to tool wear'])
                k_j = NP * process_time / tst
                k_j = math.ceil(k_j)  # 每个特征加工所需要的刀头数量
            CMU += k_j * t.at[combination[feature_set[each_f]][4], 'CST Accessory price']  # 刀头价格
            tmp_m += NP * quinary.at[combination[feature_set[each_f]][0], 'TJ Processing duration']
            tmh_m += TT1 * k_j
        for idx in range(1, len(feature_set)):
            if feature_set[idx] != feature_set[idx-1]:
                tmh_m += NP * TT
                TS += TT
        TMP.append(tmp_m)
        TMA.append(tmh_m+tmp_m)
    for mach in machine:
        NML += m.at[mach - 1, 'NMS minimal workspace']
    TS += NML/TV
    # 产品生产时间
    TA = max(TMA)  # 生产节拍
    T = TS + (NP-1)*TA
    # 设备负载均衡度
    W = (max(TMA)-min(TMA))/sum(TMA)/len(TMA)
    # 产品生产成本
    for index in range(len(machine)):
        CMF += (T-TMP[index]) * m.at[machine[index] - 1, 'CPW Unit time cost of waiting']
    C = CMD+CMF+CMU+NP*(NML*CD+CP)
    return [W, T, C]



