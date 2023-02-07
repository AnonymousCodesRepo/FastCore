import numpy as np
import pandas as pd
from fastcore.methods.methods_utils.euclidean import *
from tqdm import tqdm

def compDataGroupDisBrute(group, c_data, c_group_n):

    queryData = c_data
    dim = queryData.shape[1]
    n = int(queryData.shape[0])
    group_num = c_group_n

    dataGroupDis = np.empty((n, group_num))

    for i in tqdm(range(n)):
        dis = euclidean_dist_np(queryData[i,:].reshape(1, -1), c_data)
        df = pd.DataFrame({'dis': dis.reshape(-1), 'group': group})
        groupbyMax = df.groupby('group').max()
        groupbyMax = groupbyMax.dis.values
        dataGroupDis[i,:] = groupbyMax

    return dataGroupDis