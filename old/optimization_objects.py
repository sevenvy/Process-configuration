import math

# 直接设定：NP, TSH, TSH1, CPM, CAZ, CSV, CSV1, TVD, TVR, TV, CZJ, CD, CPR
# 读取：J, M, T, F, AD, TJ, CJ
# 由决策变量获得：m1_n, m2_j, t1_j, f1_j, a_nr, b_nks, c_nuv,
# 决策变量：i1_j, NM_n, PO_jp
# 在决策变量基础上进一步选择或优化计算：NM_n, DS


'''通过决策变量取值得到各项关联参数'''
# i1_j 各加工特征所选的五元组合编号；Q 五元组合集；m1_n 所选设备中不同型号的编号；m2_j 各加工特征所选的设备编号；t1_j 各加工特征所选的刀具编号；f1_j 各加工特征所选的夹具编号；
# a_nr 采用不同型号的设备加工的加工特征的编号；b_nks 不同型号的设备使用不同型号刀具加工的加工特征的编号；c_nuv 不同型号的设备使用不同型号夹具加工的各加工特征的编号
def variable_determine(i1_j, Q):
    m2_j = []
    t1_j = []
    f1_j = []
    m1_n = []
    for i in i1_j: # 由于Q中特征、工艺、设备等各元素编号是从1开始的，为便于后续计算，在此均减1
        m2_j.append(Q[i][2]-1)
        t1_j.append(Q[i][3]-1)
        f1_j.append(Q[i][4]-1)
        if (Q[i][2]-1) not in m1_n:
            m1_n.append(Q[i][2]-1)

    a_nr = []
    for n in range(len(m1_n)):
        a_r = []
        for j in range(len(i1_j)):
            if m1_n[n] == m2_j[j]:
                a_r.append(j)
        a_nr.append(a_r)

    b_nks = []
    for n in range(len(m1_n)):
        b_nk = [] # 记录同一设备上使用的刀具编号
        for j in a_nr[n]:
            if t1_j[j] not in b_nk:
                b_nk.append(t1_j[j])
        
        b_ks = []
        for k in range(len(b_nk)):
            b_s = []
            for j in a_nr[n]:
                if b_nk[k] == t1_j[j]:
                    b_s.append(j)
            b_ks.append(b_s)
        b_nks.append(b_ks)

    c_nuv = []
    for n in range(len(m1_n)):
        c_nu = [] # 记录同一设备上使用的夹具编号
        for j in a_nr[n]:
            if f1_j[j] not in c_nu:
                c_nu.append(f1_j[j])
        
        c_uv = []
        for u in range(len(c_uv)):
            c_v = []
            for j in a_nr[n]:
                if c_nu[k] == f1_j[j]:
                    c_v.append(j)
            c_uv.append(c_v)
        c_nuv.append(c_uv)  

    return m1_n, m2_j, t1_j, f1_j, a_nr, b_nks, c_nuv


'''系统组建成本计算'''
# NP 待加工产品数量；NM_n 不同型号设备的数量；m1_n 所选设备中不同型号的编号；
# t1_j 各加工特征所选的刀具编号；f1_j 各加工特征所选的夹具编号；i1_j 各加工特征所选的五元组合编号；
# b_nks 不同型号的设备使用不同型号刀具加工的各加工特征的编号；c_nuv 不同型号的设备使用不同型号夹具加工的各加工特征的编号；
# M 设备集；T 刀具集；F 夹具集；
# TJ 五元组合的加工时间；TSH 一次刀具安装时间；CPM 刀具安装或磨损更换的单位时间成本；CAZ 夹具安装的单位时间成本;
# CSV 物料运输系统采购单价；CSV1 物料运输系统安装调试单价
def CB_calculation(NP, NM_n, m1_n, t1_j, f1_j, i1_j, b_nks, c_nuv, M, T, F, TJ, TSH, CPM, CAZ, CSV, CSV1): 
    # 设备采购成本CTM
    CTM = sum(M[m1_n[n]][3] * NM_n[n] for n in range(len(NM_n)))
    
    # 设备安装调试成本CMC
    CMC = sum(M[m1_n[n]][4] * NM_n[n] for n in range(len(NM_n)))

    # 刀具采购成本CTT
    CTT = 0
    for n in range(len(NM_n)):
        K_num = len(b_nks[n])
        for k in range(K_num):
            S_num = len(b_nks[n][k])
            if T[t1_j[b_nks[n][k][0]]][4] != 'max': # 排除无需磨损换刀的情况
                NT_nk = NM_n[n] * math.ceil(NP * sum(TJ[i1_j[b_nks[n][k][s]]] for s in range(S_num)) / (T[t1_j[b_nks[n][k][0]]][4] * NM_n[n])) # 不同型号的设备所使用的不同型号刀具的易耗配件采购数量
                CTT += T[t1_j[b_nks[n][k][0]]][2] + NT_nk * T[t1_j[b_nks[n][k][0]]][3]

    # 刀具安装成本CTC
    times = 0
    for n in range(len(NM_n)):
        if M[m1_n[n]][1] != [8]: # 排除无需安装刀具的情况
            times += len(b_nks[n]) * NM_n[n]
    CTC = TSH * CPM * times
    
    # 夹具采购成本CTF
    CTF = 0
    for n in range(len(NM_n)):
        A = 0
        for u in range(len(c_nuv[n])):
            A += F[f1_j[c_nuv[n][u][0]]][1]
        CTF += NM_n[n] * A
    
    # 夹具安装成本CFC
    CFC = 0
    for n in range(len(NM_n)):
        B = 0
        for u in range(len(c_nuv[n])):
            B += F[f1_j[c_nuv[n][u][0]]][2]
        CFC += NM_n[n] * B
    CFC = CAZ * CFC

    # 物料运输系统采购成本CTV
    CTV = CSV * sum(NM_n)
    
    # 物料运输系统安装调试成本CVC
    CVC = CSV1 * sum(NM_n)

    # 总成本CB
    CB = CTM + CMC + CTT + CTC + CTF + CFC + CTV + CVC

    return CB


'''系统组建时间计算'''
# NM_n 不同型号设备的数量；m1_n 所选设备中不同型号的编号；
# t1_j 各加工特征所选的刀具编号；f1_j 各加工特征所选的夹具编号；
# b_nks 不同型号的设备使用不同型号刀具加工的各加工特征的编号；c_nuv 不同型号的设备使用不同型号夹具加工的各加工特征的编号；
# M 设备集；T 刀具集；F 夹具集；
# TSH 一次刀具安装时间；TVD 物料运输系统的交货期；TVR 物料运输系统安装调试时间与加工设备数量间的比例系数
def TB_calculation(NM_n, m1_n, t1_j, f1_j, b_nks, c_nuv, M, T, F, TSH, TVD, TVR):
    # 不同型号设备从开始系统组建到完成设备、刀具和夹具的安装调试所需时间TDS
    TDS = []
    for n in range(len(NM_n)):
        A = []
        B = []
        for k in range(len(b_nks[n])):
            A.append(T[t1_j[b_nks[n][k][0]]][5])
            if T[t1_j[b_nks[n][k][0]]][1] == 0:
                B.append(0)
            else:
                B.append(TSH)
        for u in range(len(c_nuv[n])):
            A.append(F[f1_j[c_nuv[n][u][0]]][4])
            B.append(F[f1_j[c_nuv[n][u][0]]][2])

        TDS_n = M[m1_n[n]][10] + M[m1_n[n]][5]
        for z in range(len(b_nks[n]) + len(c_nuv[n])):
            w = A.index(min(A))
            TDS_n = max(TDS_n, A[w]) + B[w]
            A.remove(A[w])
            B.remove(B[w])

        TDS.append(TDS_n)

    TVC = TVR * sum(NM_n)

    # 总时间TB
    TB = max(max(TDS), TVD) + TVC

    return TB


'''产品生产成本计算'''
# NP 待加工产品数量；NM_n 不同型号设备的数量；
# m1_n 所选设备中不同型号的编号；m2_j 各加工特征所选的设备编号；t1_j 各加工特征所选的刀具编号；f1_j 各加工特征所选的夹具编号；i1_j 各加工特征所选的五元组合编号；
# a_nr 采用不同型号的设备加工的各加工特征的编号；b_nks 不同型号的设备使用不同型号刀具加工的各加工特征的编号；
# J 特征集；M 设备集；T 刀具集；F 夹具集；
# TJ 五元组合的加工时间；CJ 五元组合的加工成本；AD 五元组合对应的可选刀轴方向的集合；
# TSH1 一次刀具磨损更换时间；CPM 刀具安装或磨损更换的单位时间成本；CZJ 工件装夹的单位时间成本；
# TTP 产品生产时间；CD 单位运输距离的物料运输成本；DS 生产每件产品的物料运输距离；CPR 单件产品的原材料成本
def CTP_calculation(NP, NM_n, m1_n, m2_j, t1_j, f1_j, i1_j, a_nr, b_nks, J, M, T, F, TJ, CJ, AD, TSH1, CPM, CZJ, TTP, CD, DS, CPR):
    # 工序加工成本CWJ
    CWJ = NP * sum(CJ[i1_j[j]] for j in range(len(J)))

    # 设备待机成本CMW
    CMW = 0
    for n in range(len(NM_n)):
        A = 0
        for r in range(len(a_nr[n])):
            A += TJ[i1_j[a_nr[n][r]]]
        CMW += M[m1_n[n]][6] * NM_n[n] * (TTP - NP / NM_n[n] * A)

    # 换刀成本CJH
    CJH = 0
    for n in range(len(NM_n)):
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

        CJH += M[m1_n[n]][7] * (NM_n[n] * B + (NP - NM_n[n]) * C)

    # 刀具磨损更换成本CMH
    D = 0
    for n in range(len(a_nr)):
        E = 0
        for k in range(len(b_nks[n])):
            S_num = len(b_nks[n][k])
            if T[t1_j[b_nks[n][k][0]]][4] != 'max':
                NT_nk = NM_n[n] * math.ceil(NP * sum(TJ[i1_j[b_nks[n][k][s]]] for s in range(S_num)) / (T[t1_j[b_nks[n][k][0]]][4] * NM_n[n])) # 不同型号的设备所使用的不同型号刀具的易耗配件采购数量
                E += NT_nk - NM_n[n]
                D += E
    CMH = CPM * TSH1 * D

    # 工件装夹成本CWF
    NZ = []
    for j in range(len(J)):
        if j >= 1 and m2_j[j] == m2_j[j-1] and f1_j[j] == f1_j[j-1] and ((set(AD[i1_j[j]]) & set(AD[i1_j[j-1]])) != set() or M[m2_j[j]][9] == 1):
            NZ.append(0)
        else:
            NZ.append(1)

    CWF = CZJ * NP * sum(F[f1_j[j]][3] * NZ[j] for j in range(len(J)))

    # 物料运输成本CWT
    CWT = NP * CD * DS

    # 原材料成本CRM
    CRM = NP * CPR

    CTP = CWJ + CMW + CJH + CMH + CWF + CWT + CRM 

    return CTP


'''产品生产时间计算'''
# NP 待加工产品数量；NM_n 不同型号设备的数量；
# m1_n 所选设备中不同型号的编号；m2_j 各加工特征所选的设备编号；t1_j 各加工特征所选的刀具编号；f1_j 各加工特征所选的夹具编号；i1_j 各加工特征所选的五元组合编号；
# a_nr 采用不同型号的设备加工的各加工特征的编号；b_nks 不同型号的设备使用不同型号刀具加工的各加工特征的编号；
# J 特征集；M 设备集；T 刀具集；F 夹具集；
# TJ 五元组合的加工时间；AD 五元组合对应的可选刀轴方向的集合；
# TSH1 一次刀具磨损更换时间；DS 生产每件产品的物料运输距离；TV 物料运输的平均运输速度
def TTP_calculation(NP, NM_n, m1_n, m2_j, t1_j, f1_j, i1_j, a_nr, b_nks, J, M, T, F, TJ, AD, TSH1, DS, TV):
    # 单件产品的生产时间TSP
    A = sum(TJ[i1_j[j]] for j in range(len(J)))
    B = 0
    for n in range(len(NM_n)):
        C = 0
        for r in range(len(a_nr[n])):
            if r == 0 or (r >= 1 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][r-1]]):
                NH_nr = 0
            else:
                NH_nr = 1
            C += NH_nr
        B += M[m1_n[n]][8] * C

    NZ = []
    for j in range(len(J)):
        if j >= 1 and m2_j[j] == m2_j[j-1] and f1_j[j] == f1_j[j-1] and ((set(AD[i1_j[j]]) & set(AD[i1_j[j-1]])) != set() or M[m2_j[j]][9] == 1):
            NZ.append(0)
        else:
            NZ.append(1)

    D = sum(F[f1_j[j]][3] * NZ[j] for j in range(len(J)))
    TSP = A + B + D + DS / TV

    # 不同型号的设备上生产流程的平均时间（除加工的第一件产品外）
    TMP = []
    for n in range(len(NM_n)):
        E = (NP - 1) * sum(TJ[i1_j[a_nr[n][r]]] for r in range(len(a_nr[n])))
        G = 0
        H = 0

        for r in range(len(a_nr[n])):
            if r == 0 or (r >= 1 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][r-1]]):
                NH_nr = 0
            else:
                NH_nr = 1
            G += NH_nr

            if r >= 1 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][r-1]]:
                NH1_nr = 0
            elif r == 0 and t1_j[a_nr[n][r]] == t1_j[a_nr[n][len(a_nr[n])-1]]:
                NH1_nr = 0
            else:
                NH1_nr = 1
            H += NH1_nr

        G = (NM_n[n] - 1) * G * M[m1_n[n]][8]
        H = (NP - NM_n[n]) * H * M[m1_n[n]][8]
        
        I = 0
        for k in range(len(b_nks[n])):
            S_num = len(b_nks[n][k])
            if T[t1_j[b_nks[n][k][0]]][4] != 'max':
                NT_nk = NM_n[n] * math.ceil(NP * sum(TJ[i1_j[b_nks[n][k][s]]] for s in range(S_num)) / (T[t1_j[b_nks[n][k][0]]][4] * NM_n[n])) # 不同型号的设备所使用的不同型号刀具的易耗配件采购数量
                I += NT_nk - NM_n[n]

        I = TSH1 * I
        K = (NP - 1) * sum(F[f1_j[a_nr[n][r]]][3] * NZ[a_nr[n][r]] for r in range(len(a_nr[n])))
        TMP.append((E + G + H + I + K) / (NM_n[n] * (NP - 1)))

    # 生产节拍TA
    TA = max(TMP)

    TTP = TSP + (NP - 1) * TA

    return TTP







