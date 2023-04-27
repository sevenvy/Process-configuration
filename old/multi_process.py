import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool, Pipe

# 多进程/线程设置请参考： https://blog.csdn.net/pro1515151515/article/details/125463873

#=============================================================#
# 接口                                                        #
#-------------------------------------------------------------#
#   multi_process_exec 多进程执行                             #
#   multi_thread_exec  多线程执行                             #
#-------------------------------------------------------------#
# 参数：                                                      #
#   f         (function): 批量执行的函数                      #
#   args_mat  (list)    : 批量执行的参数                      #
#   pool_size (int)     : 进程/线程池的大小                   #
#   desc      (str)     : 进度条的描述文字                    #
#-------------------------------------------------------------#
# 例子：                                                      #
# >>> def Pow(a,n):        ← 定义一个函数（可以有多个参数）   #
# ...     return a**n                                         #
# >>>                                                         #
# >>> args_mat=[[2,1],     ← 批量计算 Pow(2,1)                #
# ...           [2,2],                Pow(2,2)                #
# ...           [2,3],                Pow(2,3)                #
# ...           [2,4],                Pow(2,4)                #
# ...           [2,5],                Pow(2,5)                #
# ...           [2,6]]                Pow(2,6)                #
# >>>                                                         #
# >>> results=multi_thread_exec(Pow,args_mat,desc='计算中')   #
# 计算中: 100%|█████████████| 6/6 [00:00<00:00, 20610.83it/s] #
# >>>                                                         #
# >>> print(results)                                          #
# [2, 4, 8, 16, 32, 64]                                       #
#-------------------------------------------------------------#


'''批执行函数'''
def batch_exec(f,args_batch,w):
    results=[]
    for _,args in enumerate(args_batch):
        try:
            ans = f(*args)
            results.append(ans)
        except Exception:
            results.append(None)
        w.send(1)
    return results


'''多进程方法'''
def multi_process_exec(f,args_mat,pool_size,desc):
    if len(args_mat)==0:return []
    batch_size=max(1,int(len(args_mat)/4/pool_size)) # batch_size=max(1,int(len(args_mat)/4/pool_size))
    results=[]
    ToBatch = lambda arr,size:[arr[i*size:(i+1)*size] for i in range((size-1+len(arr))//size)]
    args_batches = ToBatch(args_mat,batch_size)

    # 显示进度条
    # with tqdm(total=len(args_mat), desc=desc) as pbar:
    #     with Pool(processes=pool_size) as pool:
    #         r,w=Pipe(duplex=False)
    #         pool_rets=[]
    #         for _,args_batch in enumerate(args_batches):
    #             pool_rets.append(pool.apply_async(batch_exec,(f,args_batch,w)))
    #         cnt=0
    #         while cnt<len(args_mat):
    #             try:
    #                 msg=r.recv()
    #                 pbar.update(1)
    #                 cnt+=1
    #             except EOFError:
    #                 break
    #         for ret in pool_rets:
    #             for r in ret.get():
    #                 results.append(r)

    # 不显示进度条
    with Pool(processes=pool_size) as pool:
        r,w=Pipe(duplex=False)
        pool_rets=[]
        for _,args_batch in enumerate(args_batches):
            pool_rets.append(pool.apply_async(batch_exec,(f,args_batch,w)))
        cnt=0
        while cnt<len(args_mat):
            try:
                msg=r.recv()
                cnt+=1
            except EOFError:
                break
        for ret in pool_rets:
            for r in ret.get():
                results.append(r)
    
    return results


'''多线程方法'''
def multi_thread_exec(f,args_mat,pool_size,desc):
    if len(args_mat)==0:return []
    results=[None for _ in range(len(args_mat))]

    # 显示进度条
    # with tqdm(total=len(args_mat), desc=desc) as pbar:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
    #         futures = {executor.submit(f,*args): i for i,args in enumerate(args_mat)}
    #         for future in concurrent.futures.as_completed(futures):
    #             i=futures[future]
    #             ret = future.result()
    #             results[i]=ret
    #             pbar.update(1)
    
    # 不显示进度条
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = {executor.submit(f,*args): i for i,args in enumerate(args_mat)}
        for future in concurrent.futures.as_completed(futures):
            i=futures[future]
            ret = future.result()
            results[i]=ret
    
    return results


# 测试函数
# def Pow(a,n):
#     return a*n

# def Pow2(T):
#     args_mat=[(2,i) for i in range(T)]
#     print(1)
#     results = multi_thread_exec(Pow,args_mat,4,desc='多进程方法1')
#     print(2)
#     return results


# if __name__=='__main__':
#     args_mat=[(2,i) for i in range(100)]
#     # results=multi_thread_exec(Pow,args_mat,1,desc='多线程')
#     # print(results)
#     args_mat1 = [[10], [10]]
    # results=multi_process_exec(Pow,args_mat,5,desc='多进程方法1')
    # print(results)


# 在终端输入 python multi_process.py 运行本程序