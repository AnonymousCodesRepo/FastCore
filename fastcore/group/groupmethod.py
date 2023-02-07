class GroupMethod(object):

    def __init__(self, dst_train, args, groupNum=1, random_seed=None, **kwargs):
        if groupNum <= 0 or groupNum > len(dst_train):
            raise ValueError("Illegal Group Number.")
        self.dst_train = dst_train
        self.num_classes = len(dst_train.classes)
        self.groupNum = groupNum
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)

    def group(self, **kwargs):
        return

