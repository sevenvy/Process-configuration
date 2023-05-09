import os
import numpy as np
import pandas as pd
from algrithm_compare import hypervolume, front_read, DPO, ENS_3
from Cluster import first_edge


# 规范化处理
# 定义获取最大最小目标值函数
def min_max(object_front):  # 以最佳前沿的各目标值集合作为输入
    min_objects = []
    max_objects = []
    for k in range(len(object_front[0])):
        min_objects.append(min([a[k] for a in object_front]))
        max_objects.append(max([a[k] for a in object_front]))
    return min_objects, max_objects


# 定义规范化处理函数
def object_normalize(object_front, min_overall, max_overall):  # 以最佳前沿的各目标值集合、所有前沿的各目标的最小值和最大值集合作为输入
    object_front_nor = [[None for _ in range(len(object_front[0]))] for _ in range(len(object_front))]
    for k in range(len(object_front[0])):
        delta = max_overall[k] - min_overall[k]
        for i in range(len(object_front)):
            object_front_nor[i][k] = (object_front[i][k] - min_overall[k]) / delta
    return object_front_nor


# 定义大小反转函数（每个目标值都用1减一下，用于求超体积）
def object_reverse(object_nor):
    object_nor_re = [[None for _ in range(len(object_nor[0]))] for _ in range(len(object_nor))]
    for k in range(len(object_nor)):
        for l in range(len(object_nor[k])):
            object_nor_re[k][l] = 1 - object_nor[k][l]
    return object_nor_re


"""多次计算同时观察各项评估指标"""
test_num_total = 20  # 一共四次计算
DPO_result = pd.DataFrame(columns=['AMOSA', 'NSGAIII', 'MOPSO', 'MOAPSO', 'NSGANS'])
for i in range(1, test_num_total+1):
    algo_hypervolume = []  # 用于记录不同算法的超体积
    algo_dpo = []  # 记录不同算法的最终解不被支配的比例
    '''读取多目标函数值数据'''
    # object_AMOSA
    AMOSA_path = os.path.join('../result', 'AMOSA', 'edge_result_' + str(i) + '.csv')
    object_AMOSA = front_read(AMOSA_path)
    # object_NSGAIII
    object_NSGA3_path = os.path.join('../result', 'NSGAIII', 'edge_result_' + str(i) + '.csv')
    object_NSGA3 = front_read(object_NSGA3_path)
    # object_MOPSO
    object_MOPSO_path = os.path.join('../result', 'MOPSO', 'edge_result_' + str(i) + '.csv')
    object_MOPSO = front_read(object_MOPSO_path)
    # object_MOAPSO
    object_MOAPSO_path = os.path.join('../result', 'MOAPSO', 'edge_result_' + str(i) + '.csv')
    object_MOAPSO = front_read(object_MOAPSO_path)
    # object_NSGA_NS
    object_NSGA_NS_path = os.path.join('../result', 'NSGA-NS', 'edge_result_' + str(i) + '.csv')
    object_NSGA_NS = front_read(object_NSGA_NS_path)
    '''求各优化目标方向上的全局极值'''
    objects_overall = object_AMOSA + object_NSGA3 + object_MOPSO + object_MOAPSO + object_NSGA_NS
    '''超体积计算'''
    min_ksi = []
    max_ksi = []
    for k in range(len(objects_overall)):
        min_ks, max_ks = min_max(objects_overall[k])
        min_ksi.append(min_ks)
        max_ksi.append(max_ks)
    min_overall = []
    max_overall = []
    for k in range(len(min_ksi[0])):
        min_overall.append(min([a[k] for a in min_ksi]))
        max_overall.append(max([a[k] for a in max_ksi]))
    '''规范化处理'''
    # AMOSA_normalizes
    AMOSA_normalizes = []
    for k in range(len(object_AMOSA)):
        AMOSA_normalizes.append(object_normalize(object_AMOSA[k], min_overall, max_overall))
    # NSGAIII_normalizes
    NSGA3_normalizes = []
    for k in range(len(object_NSGA3)):
        NSGA3_normalizes.append(object_normalize(object_NSGA3[k], min_overall, max_overall))
    # MOPSO_normalizes
    MOPSO_normalizes = []
    for k in range(len(object_MOPSO)):
        MOPSO_normalizes.append(object_normalize(object_MOPSO[k], min_overall, max_overall))
    # MOAPSO_normalizes
    MOAPSO_normalizes = []
    for k in range(len(object_MOAPSO)):
        MOAPSO_normalizes.append(object_normalize(object_MOAPSO[k], min_overall, max_overall))
    # NSGA_NS_normalizes
    NSGANS_normalizes = []
    for k in range(len(object_NSGA_NS)):
        NSGANS_normalizes.append(object_normalize(object_NSGA_NS[k], min_overall, max_overall))
    '''大小反转（原本各指标越小越好，但为便于求超体积，这里反转成越大越好）'''
    # AMOSA_reverses
    AMOSA_reverses = []
    for k in range(len(AMOSA_normalizes)):
        AMOSA_reverses.append(object_reverse(AMOSA_normalizes[k]))
    # NSGAIII_reverses
    NSGA3_reverses = []
    for k in range(len(NSGA3_normalizes)):
        NSGA3_reverses.append(object_reverse(NSGA3_normalizes[k]))
    # MOPSO_reverses
    MOPSO_reverses = []
    for k in range(len(MOPSO_normalizes)):
        MOPSO_reverses.append(object_reverse(MOPSO_normalizes[k]))
    # MOAPSO_reverses
    MOAPSO_reverses = []
    for k in range(len(MOAPSO_normalizes)):
        MOAPSO_reverses.append(object_reverse(MOAPSO_normalizes[k]))
    # NSGA_NS_reverses
    NSGA_NS_reverses = []
    for k in range(len(NSGANS_normalizes)):
        NSGA_NS_reverses.append(object_reverse(NSGANS_normalizes[k]))
    '''超体积比较(越大越好)'''
    # AMOSA_volumes
    AMOSA_volume = []
    for k in range(len(AMOSA_reverses)):
        AMOSA_volume.append(hypervolume(AMOSA_reverses[k]))
    # NSGA3_volumes
    NSGA3_volume = []
    for k in range(len(NSGA3_reverses)):
        NSGA3_volume.append(hypervolume(NSGA3_reverses[k]))
    # MOPSO_volumes
    MOPSO_volume = []
    for k in range(len(MOPSO_reverses)):
        MOPSO_volume.append(hypervolume(MOPSO_reverses[k]))
    # MOAPSO_volumes
    MOAPSO_volume = []
    for k in range(len(MOAPSO_reverses)):
        MOAPSO_volume.append(hypervolume(MOAPSO_reverses[k]))
    # NSGA_NS_volumes
    NSGA_NS_volume = []
    for k in range(len(NSGA_NS_reverses)):
        NSGA_NS_volume.append(hypervolume(NSGA_NS_reverses[k]))
    '''记录不同算法的超体积数据'''
    algo_hypervolume.append(AMOSA_volume)
    algo_hypervolume.append(NSGA3_volume)
    algo_hypervolume.append(MOPSO_volume)
    algo_hypervolume.append(MOAPSO_volume)
    algo_hypervolume.append(NSGA_NS_volume)
    # 将列表转置，结果写入csv文件
    algo_hypervolume = np.array(algo_hypervolume)
    algo_hypervolume = algo_hypervolume.T
    algo_hypervolume = pd.DataFrame(algo_hypervolume)
    algo_hypervolume.columns = ['AMOSA', 'NSGAIII', 'MOPSO', 'MOAPSO', 'NSGANS']  # 为每一列数据添加标签
    algo_hypervolume.to_csv(os.path.join('../result', 'hypervolume', 'test_' + str(i) + '.csv'), index=False)

    '''支配比例'''
    front_overall = object_AMOSA[-1] + object_NSGA3[-1] + object_MOPSO[-1] + object_MOAPSO[-1] + object_NSGA_NS[-1]
    AMOSA_DPO_num = DPO(front_overall, object_AMOSA[-1])
    NSGA3_DPO_num = DPO(front_overall, object_NSGA3[-1])
    MOPSO_DPO_num = DPO(front_overall, object_MOPSO[-1])
    MOAPSO_DPO_num = DPO(front_overall, object_MOAPSO[-1])
    NSGANS_DPO_num = DPO(front_overall, object_NSGA_NS[-1])
    # 计算比例
    AMOSA_DPO = AMOSA_DPO_num / len(object_AMOSA[-1])
    algo_dpo.append(AMOSA_DPO)
    NSGA3_DPO = NSGA3_DPO_num / len(object_NSGA3[-1])
    algo_dpo.append(NSGA3_DPO)
    MOPSO_DPO = MOPSO_DPO_num / len(object_MOPSO[-1])
    algo_dpo.append(MOPSO_DPO)
    MOAPSO_DPO = MOAPSO_DPO_num / len(object_MOAPSO[-1])
    algo_dpo.append(MOAPSO_DPO)
    NSGANS_DPO = NSGANS_DPO_num / len(object_NSGA_NS[-1])
    algo_dpo.append(NSGANS_DPO)
    algo_dpo = pd.DataFrame([algo_dpo], columns=['AMOSA', 'NSGAIII', 'MOPSO', 'MOAPSO', 'NSGANS'])
    DPO_result = DPO_result.append(algo_dpo)
    print(DPO_result)
    print('第' + str(i) + '次计算完成！')
DPO_result.to_csv('../result/DPO/DPO.csv', index=False)


