import numpy as np


     
def calc_weights(distance_matrix, group_siz):
    min_sample = np.argmin(distance_matrix, axis=0)      
    weights = np.zeros(distance_matrix.shape[0])

    for i, group_n in zip(min_sample, group_siz):
        weights[i] = weights[i] + group_n
    return weights