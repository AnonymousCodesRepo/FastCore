import numpy as np
from .coresetmethod import CoresetMethod
 
import fastcore.group as groupMethods
from .methods_utils.euclidean import *
import pandas as pd
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import faiss
from .scale_utils.distance import *
from .scale_utils import convHull
from .scale_utils import PQ as PQutils
from .scale_utils.coreWeight import *
from .scale_utils.preprocessBrute import *


class Scale(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, groupNum=1000, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.n_train = len(dst_train)
        self.groupNum = groupNum

        epsilon = 0.05
        self.sample_size = np.int(1.0/fraction * np.log(1 / epsilon))
        if self.sample_size > 500:
            self.sample_size = 500
         


        self.group_method = groupMethods.__dict__[args.groupAlg](self.dst_train, args=args, groupNum=self.groupNum,  random_seed=1234)
        self.groups = self.group_method.group()


        self.distanceAlg = args.distAlg     
        self.m = args.PQm                    
        self.nbits = args.PQnbits            
        self.verbose = True if args.verbose > 0 else False

     
    def getClassInfo(self, c):
        c_index = np.argwhere(self.dst_train.targets == c).reshape(-1)

        c_n = c_index.shape[0]                       
        c_data = self.dst_train.data[c_index, :]     
        c_group = self.groups[c_index]               
        c_group_n = c_group.max() + 1                
        c_group_siz = np.bincount(c_group)           
        c_core_n = np.int(self.fraction * c_n)       
        dim = c_data.shape[1]
        CSGroupDis = np.ones(c_group_n, dtype=np.int64) * 999999999              
        CSdataGroupDis = np.empty((c_core_n, c_group_n), dtype=np.float64)         

        c_data_available = np.arange(c_n, dtype=np.int32)        

        print(f"Class [{c}]   c_n [{c_n}]   c_group_n [{c_group_n}]   c_core_n [{c_core_n}] c_dim [{dim}]")
        return c_index, c_n, c_data, c_group, c_group_n, c_core_n, c_group_siz, CSGroupDis, CSdataGroupDis, c_data_available



    def computeUtility(self, queryData, data, group, CSGroupDis, queryIdx=None):


        queryGroupDis = computeDis(self, queryData, data, group, queryIdx)    
        tmp_dis = np.concatenate((queryGroupDis.reshape(-1, 1), CSGroupDis.reshape(-1, 1)), axis=1)
        tmp_dis = tmp_dis.min(axis=1)
         
        return -tmp_dis.sum()


    def vec_computeUtility(self, sample_data, sample_idxs, CSGroupDis, c_data=None, c_group=None):

        if self.distanceAlg not in ["pqSDC", "brute", "convex"]:
            assert c_data is not None
            assert c_group is not None
            utilities = [self.computeUtility(sample_data[j,:], c_data, c_group, CSGroupDis) for j in range(sample_data.shape[0])]

        else:

            if self.dataGroupDis is not None:
                 
                dis = self.dataGroupDis[sample_idxs, :]
            else:
                print("Maybe compute utility from scratch!")
                dis = computeDis(self, sample_data, c_data, c_group, queryIdx=sample_idxs, plainGroup=(self.args.groupAlg=="Plain"))

            tmp = np.stack((dis, np.repeat(np.expand_dims(CSGroupDis, 0), sample_idxs.shape[0], axis=0)))
            distances = tmp.min(axis=0)
            utilities = - distances.sum(axis=1)
             
             
        return utilities, distances

    def needsHull(self):

        return self.distanceAlg == "convex"

    def needsPQ(self):

        return self.distanceAlg in ["pqSDC", "pqADC", "brutePQ"]

    def prepareForHull(self, c_group, c_group_n, c_data, precompute=True):

        self.hull, self.g_hull = convHull.computeHull(c_group, c_group_n, c_data)
        if precompute == True:
            self.dataGroupDis= convHull.compDataGroupDisConvex(self.hull, self.g_hull, c_data, c_group_n)
        return

    def prepareForPQ(self,c_group, c_group_n, c_data, precompute=True):

        c_n = c_data.shape[0]

        if c_n < (2**self.nbits):
            tmp = np.floor(np.log2(c_n))
            if tmp >=2:
                self.ipq, self.pq, self.codes = PQutils.buildPQ(self.m, 2, c_data)   
            else:
                self.ipq, self.pq, self.codes = PQutils.buildPQ(self.m, 1, c_data)   
        else:
            self.ipq, self.pq, self.codes = PQutils.buildPQ(self.m, self.nbits, c_data)      
        if self.distanceAlg == "pqSDC":
            self.pq.compute_sdc_table()
            self.groupPQids = PQutils.computeGroupPQids(self.m, self.codes, c_group)         

             
            sdc_tab = faiss.vector_to_array(self.pq.sdc_table)
            self.sdc_tab = sdc_tab.reshape(self.m, self.pq.ksub, self.pq.ksub)
            self.groupPQdis = PQutils.precomputeGroupPQDis(self.m, c_group_n, self.sdc_tab, self.groupPQids, pq=self.pq)

        if precompute == False:
            return

        print(f"Precompute data-group dis for {self.distanceAlg}")
         
        if self.distanceAlg == "pqADC":
            self.dataGroupDis = PQutils.precompDataGroupDisADC(c_group, c_data, self.ipq, self.pq, self.groupPQids)

         
        elif self.distanceAlg == "pqSDC":
            self.dataGroupDis = PQutils.precompDataGroupDisSDC(self.m, self.codes, c_group, c_data, self.ipq, self.pq,
                                                        self.groupPQids, self.groupPQdis, self.sdc_tab)
        return

    def prepare(self, c_group, c_group_n, c_data, precompute=True):

        self.dataGroupDis = None

        if self.distanceAlg == "brute":
            c_n = c_data.shape[0]
             
            if c_n < 10**5 and precompute==True:
                print(f"Prepare for brute    precompute data-group dis [{precompute}]")
                self.dataGroupDis = compDataGroupDisBrute(c_group, c_data, c_group_n)
        if self.needsHull() == True:
            print(f"Prepare for convex hull    precompute data-group dis [{precompute}]")
            self.prepareForHull(c_group, c_group_n, c_data, precompute)
        if self.needsPQ() == True:
            print(f"Prepare for PQ     precompute data-group dis [{precompute}]")
            self.prepareForPQ(c_group, c_group_n, c_data, precompute)

    def selectCoreset(self):
        rng = np.random.default_rng(self.random_seed)
        self.index  = np.array([], dtype=np.int32)
        self.weight = np.array([], dtype=np.int32)

         
        for c in range(self.num_classes):
             
            c_index, c_n, c_data, c_group, c_group_n, c_core_n, c_group_siz, CSGroupDis, CSdataGroupDis,  c_data_available = self.getClassInfo(c)
            if c_core_n == 0:
                continue
            c_indices = np.array([], dtype=np.int32)     

            self.prepare(c_group, c_group_n, c_data, self.args.precompute)

             
            for i in range(c_core_n):
                print("i = [{:5d}]".format(i))
                 
                if c_data_available.shape[0] > self.sample_size:
                    sample_idxs = rng.choice(c_data_available, self.sample_size, replace=False)      
                else:
                    sample_idxs = c_data_available
                sample_data = c_data[sample_idxs]                                                

                 
                st_time = time.time()
                utility, distances = self.vec_computeUtility(sample_data, sample_idxs, CSGroupDis, c_data, c_group)
                if self.verbose:
                    print("Compute utility took  ", time.time() - st_time)

                 
                choose_id = np.argmax(utility)       
                choose_idx = sample_idxs[choose_id]  
                choose_index = c_index[choose_idx]  

                print("max utility is ", utility.max())

                 
                c_indices = np.append(c_indices, choose_index)   



                selected_dis = distances[choose_id, :]


                CSdataGroupDis[i,:] = selected_dis


                tmp_dis = np.stack((CSGroupDis, selected_dis))
                CSGroupDis = np.min(tmp_dis, axis=0)            

                 
                choose_loc = np.argwhere(c_data_available==choose_idx).reshape(-1)[0]

                 
                c_data_available[choose_loc] = c_data_available[-1]
                c_data_available = c_data_available[:-1]

                if self.verbose:
                    print("Add an ele took ", time.time() - st_time)
             
            st_time = time.time()
            c_weights = calc_weights(CSdataGroupDis, c_group_siz)
            self.index = np.append(self.index, c_indices)
            self.weight = np.append(self.weight, c_weights)
            print("Compute weight took ", time.time() - st_time)

        print("!")

        non_zero_idx  = self.weight!=0
        return self.index[non_zero_idx], self.weight[non_zero_idx]
         

    def select(self, **kwargs):
        selection_result, weights = self.selectCoreset()
        return {"indices": selection_result, "weights": weights}
         