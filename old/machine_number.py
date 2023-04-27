import math


'''设备数量的可行解空间'''
# NP 待生产的产品数量；NMP 车间设备工位数量上限；
# M 设备集；T 刀具集；
# TSH1 一次刀具磨损更换时间；TJ 五元组合的加工时间；AD 五元组合对应的可选刀轴方向的集合；
# m1_n 所选设备中不同型号的编号；m2_j 各加工特征所选的设备编号；t1_j 各加工特征所选的刀具编号；f1_j 各加工特征所选的夹具编号；i1_j 各加工特征所选的五元组合编号；
# a_nr 采用不同型号的设备加工的各加工特征的编号；b_nks 不同型号的设备使用不同型号刀具加工的各加工特征的编号
def NM_calculation(NP, NMP, M, T, TSH1, TJ, AD, m1_n, m2_j, t1_j, f1_j, i1_j, a_nr, b_nks):
    E = [1 for _ in range(len(m1_n))]  # 第一个解
    SE = [E.copy()]
    e = len(m1_n)
    while e < NMP:        
        # 不同型号的设备上生产流程的平均时间（除加工的第一件产品外）
        TMP = []
        for n in range(len(m1_n)):
            A = (NP - 1) * sum(TJ[i1_j[a_nr[n][r]]] for r in range(len(a_nr[n])))      
            
            B = 0
            C = 0
            for r in range(len(a_nr[n])):
                if r == 0 or (r >= 1 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][r-1]]):
                    NH_nr = 0
                else:
                    NH_nr = 1
                B += NH_nr

                if r >= 1 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][r-1]]:
                    NH1_nr = 0
                elif r == 0 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][len(a_nr[n])-1]]:
                    NH1_nr = 0
                else:
                    NH1_nr = 1
                C += NH1_nr
            B = (E[n] - 1) * B * M[m1_n[n]][8]
            C = (NP - E[n]) * C * M[m1_n[n]][8]

            D = 0
            for k in range(len(b_nks[n])):
                S_num = len(b_nks[n][k])
                if T[t1_j[b_nks[n][k][0]]][4] != 'max':
                    NT_nk = E[n] * math.ceil(NP * sum(TJ[i1_j[b_nks[n][k][s]]] for s in range(S_num)) / (T[t1_j[b_nks[n][k][0]]][4] * E[n])) # 不同型号的设备所使用的不同型号刀具的易耗配件采购数量
                    D += NT_nk - E[n]
            D = TSH1 * D

            NZ = []
            for j in range(len(i1_j)):
                if j >= 1 and m2_j[j] == m2_j[j-1] and f1_j[j] == f1_j[j-1] and ((set(AD[i1_j[j]]) & set(AD[i1_j[j-1]])) != set() or M[m2_j[j]][9] == 1):
                    NZ.append(0)
                else:
                    NZ.append(1)

            G = (NP - 1) * sum(T[f1_j[a_nr[n][r]]][3] * NZ[a_nr[n][r]] for r in range(len(a_nr[n])))
            TMP.append((A + B + C + D + G) / (E[n] * (NP - 1)))

        # 生产节拍TA
        TA = max(TMP)

        E[TMP.index(TA)] += 1
        e += 1

        if E[TMP.index(TA)] > M[m1_n[TMP.index(TA)]][11] or E[TMP.index(TA)] > NP:
            break
        else:
            SE.append(E.copy())

    return SE

