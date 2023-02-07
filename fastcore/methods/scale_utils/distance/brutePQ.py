import faiss
import pandas as pd
import numpy as np


def computePQScan(queryData, data, group, ipq, pq):

    sp = faiss.swig_ptr

    dim = queryData.shape[1]
    m = int(dim/pq.dsub)

          

    query = queryData.astype("float32")
    dis_tab = np.zeros((1, m, pq.ksub), "float32")

    pq.compute_distance_tables(1, sp(query), sp(dis_tab))


          
    codes = faiss.vector_to_array(ipq.codes)
    codes = codes.reshape(-1, m)


          
    this_dis_tab = dis_tab[0,:]
          
    all_dis = this_dis_tab[0, codes[:,0]] + this_dis_tab[1, codes[:, 1]]


          

          

    df = pd.DataFrame({'dis': all_dis, 'group': group})
    groupbyMax = df.groupby('group').max()
    groupbyMax = groupbyMax.dis.values
    return groupbyMax