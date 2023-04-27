import random
import os
import pandas as pd


# initials 初始种群;
def generate_initials(j, q, n):
    """
    :param q: 五元组合集
    :param n: 所需的种群数量
    :return: 初始种群
    """
    initials = []
    for _ in range(n):
        initial = []
        k = 1  # 是否执行选择的标志
        while k == 1:
            for feature in range(len(j)):  # 对于每个加工特征分别随机选取五元组合
                quinary_i = q[q['J Feature'] == feature+1]
                process = quinary_i.sample(1).values[0].astype(int).tolist()[0:5]
                initial.append(process)
            if initial in initials:  # 如果随机选择的五元组合已存在，为避免重复，重新选择
                k = 1
            else:
                k = 0
        initials.append(initial)
    return initials


if __name__ == '__main__':
    N = 100  # 种群数量
    J = pd.read_csv('../database/feature.csv')
    max_run = 10  # 最大运行次数

    for run in range(1, max_run+1):
        initials = []
        for _ in range(N):
            initial = []
            k = 1  # 是否执行选择的标志
            while k == 1:
                for feature in range(len(J)):  # 对于每个加工特征分别随机选取五元组合
                    initial.append(random.random())
                if initial in initials:  # 如果随机选择的五元组合已存在，为避免重复，重新选择
                    k = 1
                else:
                    k = 0
            initials.append(initial)
        print(initials)
        pd.DataFrame(initials).to_csv(os.path.join('../database', 'initials', 'initials_'+str(run)+'.csv'), index=False)




