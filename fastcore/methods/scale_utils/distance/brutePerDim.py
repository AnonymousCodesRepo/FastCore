from fastcore.methods.methods_utils.euclidean import *
import pandas as pd


def computeDivideDis(queryData, data, group):

    dim = len(queryData.shape)
    if dim == 1:
        queryData = queryData.reshape(1, -1)
          

          
    dis_per_two_dim = []

          
    for i in range(int(data.shape[1]/2)):
        st = 2 * i
        en = 2 * (i + 1)
        dis = euclidean_dist2_np(queryData[:,st:en], data[:,st:en])
        dis_per_two_dim.append(dis)

          
    df = pd.DataFrame({'group':group})
    for i in range(int(data.shape[1]/2)):
        df[f"col{i}"] = dis_per_two_dim[i].reshape(-1)
    groupbyMax = df.groupby('group').max()
    groupbyMax = groupbyMax.sum(axis=1).values
    return groupbyMax