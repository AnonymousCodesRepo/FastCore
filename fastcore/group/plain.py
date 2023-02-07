import numpy as np
from .groupmethod import GroupMethod


class Plain(GroupMethod):
    def __init__(self, dst_train, args, groupNum, random_seed=None,  **kwargs):
        super().__init__(dst_train, args, groupNum, random_seed)
        self.n_train = len(dst_train)

    def group_balance(self):

        self.groupIndex = np.empty(self.n_train, dtype=np.int64)

        for c in range(self.num_classes):
            c_index = np.argwhere(self.dst_train.targets==c)
            c_index = c_index.reshape(-1)
            c_n = c_index.shape[0]
            self.groupIndex[c_index] = np.arange(c_n)

        return self.groupIndex

    def group(self, **kwargs):
        return self.group_balance()
