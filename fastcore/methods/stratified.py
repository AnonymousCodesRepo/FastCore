import numpy as np
from .coresetmethod import CoresetMethod
from fastcore.group import *
from .methods_utils.euclidean import *
import pandas as pd
import time



class Stratified(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, groupNum=1000, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.n_train = len(dst_train)
        self.groupNum = groupNum

        self.group_method = LSH(self.dst_train, args=args, groupNum=self.groupNum, random_seed=1234, balance=True, nbits=8)
        self.groups = self.group_method.group()

        
    def getClassInfo(self, c):
        c_index = np.argwhere(self.dst_train.targets == c)
        c_index = c_index.reshape(-1)
        c_n = c_index.shape[0]      
            
        c_data = self.dst_train.data[c_index, :]
            
        c_group = self.groups[c_index]
        c_group_n = c_group.max() + 1

            
        c_core_n = np.int(self.fraction * c_n)
        print("c = {} coreset size = ", c_core_n)


        c_group_siz = np.bincount(c_group)

        CSGroupDis = np.ones(c_group_n, dtype=np.int64) * 999999999
        CSdataGroupDis = np.empty((c_core_n, c_group_n), dtype=np.float64)

            
        c_indices = np.array([], dtype=np.int32)

            
            
            
        c_data_available = np.arange(c_n, dtype=np.int32)

        return c_index, c_n, c_data, c_group, c_group_n, c_core_n, c_group_siz, CSGroupDis, CSdataGroupDis, c_indices, c_data_available



    def select_balance(self):
        rng = np.random.default_rng(self.random_seed)
        self.index  = np.array([], dtype=np.int32)

            
        for c in range(self.num_classes):
                
            c_index, c_n, c_data, c_group, c_group_n, c_core_n, c_group_siz, CSGroupDis, CSdataGroupDis, c_indices, c_data_available = self.getClassInfo(c)

                
            print(f"c {c}")
            print()
            group_sample_siz = (c_core_n * c_group_siz / c_n + 0.5).astype(np.int32)
            for gid in range(c_group_n):
                    
                g_sample_siz = group_sample_siz[gid]

                    
                g_ids = np.argwhere(c_group == gid)         
                g_ids = c_index[g_ids]                      

                    
                g_samples = rng.choice(g_ids, g_sample_siz, replace=False)
                c_indices = np.append(c_indices, g_samples)


            self.index = np.append(self.index, c_indices)

        return self.index

    def select(self, **kwargs):
        selection_result = self.select_balance()
        return {"indices": selection_result}