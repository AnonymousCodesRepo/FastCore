import faiss
import pandas as pd
import numpy as np

def computePQGroupSum(queryData, data, group, ipq, pq, groupPQidx):
       

    sp = faiss.swig_ptr

    dim = queryData.shape[1]
    m = int(dim/pq.dsub)

       

    query = queryData.astype("float32")
    dis_tab = np.zeros((1, m, pq.ksub), "float32")

    pq.compute_distance_tables(1, sp(query), sp(dis_tab))

       
    group_num = groupPQidx.shape[0]

       
    this_dis_tab = dis_tab[0,:]

    groupbyMax = np.empty(group_num)

    for i in range(group_num):
        tmp_sum = 0
        for j in range(m):
            tmp_sum = tmp_sum + this_dis_tab[j, groupPQidx[i, j]].max()
        groupbyMax[i] = tmp_sum


       
       
       
    return groupbyMax