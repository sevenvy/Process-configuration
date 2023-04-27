import random
import os
import pandas as pd


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




