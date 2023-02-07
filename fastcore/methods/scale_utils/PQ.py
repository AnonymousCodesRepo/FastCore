import numpy as np
import pandas as pd
import time

import faiss
import torch

def buildPQ(m, nbits, c_data):
    n = c_data.shape[0]
    dim = c_data.shape[1]

         
    ipq = faiss.IndexPQ(dim, m, nbits)
    ipq.train(c_data)
    ipq.add(c_data)

    pq = ipq.pq

    centroids = faiss.vector_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)


    sp = faiss.swig_ptr

         
    print("See pq.ksub is ", pq.ksub)
    query = c_data[0, :].astype("float32").reshape(1, -1)
    new_tab = np.zeros((1, m, pq.ksub), "float32")

         
    pq.compute_distance_tables(1, sp(query), sp(new_tab))

         
    tot_size = n * m
    codes = np.empty(tot_size, dtype=np.int32)
         
    raw_codes = pq.compute_codes(c_data)
    bs = faiss.BitstringReader(faiss.swig_ptr(raw_codes[0]), raw_codes.shape[1])
    for i in range(tot_size):
        codes[i] = bs.read(nbits)

         
    codes = codes.reshape(-1, m)

    return ipq, pq, codes


     
     
def computeGroupPQids(m, codes, group):
         
    df = pd.DataFrame({'code{}'.format(i): codes[:, i].astype(np.int32) for i in range(m)})
    df['group'] = group

    groupPQidx = df.groupby('group').agg(['unique'])
    groupPQidx = groupPQidx.values
    return groupPQidx



def precompDataGroupDisADC(group, c_data, ipq, pq, groupPQids):
    sp = faiss.swig_ptr

         
    queryData = c_data
    dim = queryData.shape[1]
    m = int(dim / pq.dsub)
    n = int(queryData.shape[0])

         
    query = queryData.astype("float32")
    dis_tabs = np.zeros((n, m, pq.ksub), "float32")
    pq.compute_distance_tables(n, sp(query), sp(dis_tabs))

         
    group_num = groupPQids.shape[0]          


    dataGroupDis = np.empty((n, group_num))



    n_ids = np.arange(n)
    n_ids = n_ids.repeat(group_num * m)

    g_ids = np.arange(group_num)
    g_ids = g_ids.repeat(m)
    g_ids = np.repeat(g_ids.reshape(1, -1), n, axis=0).reshape(-1)
         

    m_ids = np.arange(m)
    m_ids = np.repeat(m_ids.reshape(1, -1), n*group_num, axis=0).reshape(-1)
    print(m_ids.shape)

    tmp = dis_tabs[n_ids, g_ids, groupPQids[g_ids, m_ids]].max()

    for k in range(n):
        for i in range(group_num):
            tmp_sum = 0
            for j in range(m):
                tmp_sum = tmp_sum + dis_tabs[k, j, groupPQids.iloc[i, j]].max()
            dataGroupDis[k, i] = tmp_sum


    return dataGroupDis

def precomputeGroupPQDis(m, c_group_n, sdc_tab, groupPQids, pq):

         
    groupPQDis = np.empty((c_group_n, m, pq.ksub))


    for i in range(c_group_n):
        for j in range(m):
            groupPQDis[i, j, :] = sdc_tab[j, groupPQids[i,j],:].max(axis=0)
    return groupPQDis



def precompDataGroupDisSDC(m, codes, group, c_data, ipq, pq, groupPQids, groupPQDis, sdc_tab):

    n = c_data.shape[0]
    c_group_n = groupPQids.shape[0]



    st_t = time.time()
         
    groupDataDis = np.empty((c_group_n, n))

         
    m_index = np.arange(m, dtype=np.int)
    m_index = np.repeat(m_index.reshape(1, -1), n, axis=0)
         
    pq_index = codes

    for g in range(c_group_n):
        tmp_arr = groupPQDis[g, m_index, pq_index]
        groupDataDis[g, :] = tmp_arr.sum(axis=1)

    print(f"Compute data-group dis took [{time.time() - st_t}]")
    maxDis = groupDataDis.T
         
    return maxDis



def computeDataGroupDisSDC(scale, queryIdx, ifplainGroup, groupPQDis=None):
    n = queryIdx.shape[0]
    m = scale.m
    codes = scale.codes
    pq = scale.pq

         
    sdc_tab = faiss.vector_to_array(pq.sdc_table)
    sdc_tab = sdc_tab.reshape(m, pq.ksub, pq.ksub)

    queryCodes = codes[queryIdx, :]
    n_query = queryIdx.shape[0]
    c_group_n = scale.groupPQids.shape[0]

         
         
    if groupPQDis is None:
        print("【Warn】 Recompute groupPQDis when --precompute==False, which should be optimized")
        groupPQDis = np.empty((c_group_n, m, pq.ksub))


        for i in range(c_group_n):
            for j in range(m):
                groupPQDis[i, j, :] = sdc_tab[j, scale.groupPQids[i,j],:].max(axis=0)


    st_t = time.time()
         
    groupDataDis = np.empty((c_group_n, n))

         
    m_index = np.arange(m, dtype=np.int)
    m_index = np.repeat(m_index.reshape(1, -1), n_query, axis=0)
         
    pq_index = queryCodes

    for g in range(c_group_n):
        tmp_arr = groupPQDis[g, m_index, pq_index]
        groupDataDis[g, :] = tmp_arr.sum(axis=1)

    print(f"Compute data-group dis took [{time.time() - st_t}]")
    maxDis = groupDataDis.T


         
    return maxDis