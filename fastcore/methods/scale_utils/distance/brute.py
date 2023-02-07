from fastcore.methods.methods_utils.euclidean import *
import pandas as pd


def computeDisBrute(queryData, data, group, plainGroup=False):

    dim = len(queryData.shape)
    if dim == 1:
        queryData = queryData.reshape(1, -1)

    # dis = euclidean_dist_np(queryData, data)
    queryData_torch = torch.as_tensor(queryData)
    data_torch = torch.as_tensor(data)
    dis = euclidean_dist2_torch(queryData_torch, data_torch).numpy()


    if plainGroup == False:

        df = pd.DataFrame({'dis':dis.reshape(-1), 'group':group})
        groupbyMax = df.groupby('group').max()
        groupbyMax = groupbyMax.dis.values
    else:
        groupbyMax = dis

    return groupbyMax

