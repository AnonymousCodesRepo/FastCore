import numpy as np
from .groupmethod import GroupMethod
import faiss

from sklearn.preprocessing import OrdinalEncoder



class LSH(GroupMethod):
    def __init__(self, dst_train, args, groupNum, random_seed=None, **kwargs):
        super().__init__(dst_train, args, groupNum, random_seed)
        self.n_train = len(dst_train)

        self.nbits = args.LSHnbits

    def group_balance(self):
        # np.random.seed(self.random_seed)
        rng = np.random.default_rng(self.random_seed)

        totalGroupNum = 0

        self.groupIndex = np.empty(self.n_train, dtype=np.int64)

        for c in range(self.num_classes):
            # c_index = (self.dst_train.targets == c)
            c_index = np.argwhere(self.dst_train.targets==c)
            c_index = c_index.reshape(-1)
            c_n = c_index.shape[0]

            c_data = self.dst_train.data[c_index, :]

            index = faiss.IndexLSH(c_data.shape[1], self.nbits)
            index.add(c_data)
            # arr = faiss.vector_to_array(index.codes).reshape(-1, 1)
            arr = faiss.vector_to_array(index.codes).reshape(c_n, -1)


            tmp = arr * np.array([2 ** (8 * i) for i in range(arr.shape[1])])
            tmp = tmp.sum(axis=1).reshape(-1, 1)


            enc = OrdinalEncoder()
            enc.fit(tmp)
            encoded_arr = enc.transform(tmp).astype(np.int64)

            print(f"Category [{c}] is divided into [{len(enc.categories_[0])}] groups")
            totalGroupNum = totalGroupNum + len(enc.categories_[0])

            self.groupIndex[c_index] = encoded_arr.reshape(-1)


        self.groupNum = totalGroupNum
        return self.groupIndex

    def group(self, **kwargs):
        return self.group_balance()
