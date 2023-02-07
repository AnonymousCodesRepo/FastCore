import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from tqdm import tqdm
from fastcore.methods.methods_utils.euclidean import *

def computeHull(c_group, c_group_n, c_data):

        
    dim = c_data.shape[1]
    n_subspace = np.int(dim / 2)


    hull = []
    g_hull = []
    for i in range(n_subspace):
        hull.append([])
        g_hull.append([])

        
    for g in range(c_group_n):
            
        cg_index = np.argwhere(c_group == g)
        cg_data = c_data[cg_index.reshape(-1)]
            
        for i in range(n_subspace):
            st = 2 * i
            en = 2 * (i + 1)
                
            if cg_data.shape[0] <=2 :
                hull_vertices = cg_data[:, st:en]
            else:
                    
                try:
                    hull_ = ConvexHull(cg_data[:, st:en])
                    hull_vertices = cg_data[hull_.vertices, st:en]
                except:
                        
                    hull_vertices = np.unique(cg_data[:, st:en], axis=0)

                
            hull[i].append(hull_vertices)
            g_hull[i].append([g] * hull_vertices.shape[0])

        
        
    for i in range(n_subspace):
        assert len(hull[i]) == c_group_n
        assert len(g_hull[i]) == c_group_n

        hull[i] = np.concatenate(hull[i], axis=0)
        g_hull[i] = np.concatenate(g_hull[i])

        assert hull[i].shape[0] == g_hull[i].shape[0]
        
    return hull, g_hull


def compDataGroupDisConvex(hull, g_hull, c_data, c_group_n):

        
    data = torch.Tensor(c_data)
    dim = data.shape[1]
    n_subspace = np.int(dim / 2)
    c_n = int(data.shape[0])

    dataGroupDis = torch.zeros((c_n, c_group_n))
    for i in tqdm(range(n_subspace)):       
        st = 2 * i
        en = 2 * (i + 1)

        subData = data[:, st:en]                    
        hull_i = torch.as_tensor(hull[i], dtype=torch.float32)           

            
        g_hull_i = torch.as_tensor(g_hull[i], dtype=torch.int64)

            
        ith_dataGroupDis = euclidean_dist2_torch(subData, hull_i)


        g_hulls = g_hull_i.repeat(c_n)
        g_hulls = g_hulls.reshape(c_n, g_hull_i.shape[0])
        dataGroupDis_i = torch.zeros_like(dataGroupDis)
        dataGroupDis_i.scatter_reduce_(src=ith_dataGroupDis, index=g_hulls, dim=1, reduce="amax", include_self=False)
        dataGroupDis = dataGroupDis + dataGroupDis_i


    return dataGroupDis.numpy()