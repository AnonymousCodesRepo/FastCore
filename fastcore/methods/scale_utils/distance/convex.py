from fastcore.methods.methods_utils.euclidean import *
import pandas as pd
import numpy as np


def computeDivideHullDis(queryData, data, group, hull, g_hull):

    n_dim = len(queryData.shape)

    if n_dim == 1:
        queryData = queryData.reshape(1, -1)

    dim = queryData.shape[1]
    n_subspace = np.int(dim / 2)


       

       
    dis_per_two_dim = []

       
    for i in range(n_subspace):
        st = 2 * i
        en = 2 * (i + 1)
        dis = euclidean_dist2_np(queryData[:,st:en], hull[i])
           
        dis_per_two_dim.append(dis)

    allDimMax = []
       
    for i in range(n_subspace):
        df = pd.DataFrame({'group':g_hull[i]})
        df['val'] = dis_per_two_dim[i].reshape(-1)
        groupbyMax = df.groupby('group').max()
        allDimMax.append(groupbyMax.values)
    allDimMax = np.concatenate(allDimMax, axis=1)
    allDimMax = allDimMax.sum(axis=1)
    return allDimMax