from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data.dataset import *
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import check_integrity
import pandas as pd



class brazil(Dataset):
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        # super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train  # training set or test set
        self.transform = None
        self.target_transform = None
        # self.classes = torch.Tensort([0,1])
        self.classes = [0, 1, 2, 3, 4]


        self.data, self.targets = self._load_data()



    def _load_data(self):

        DIR = "/home/anonymous/disk/C-craig/dataset/"
        dataName = "Brazilnew"
        train_df = pd.read_csv(DIR + "{}-train.csv".format(dataName))
        val_df = pd.read_csv(DIR + "{}-val.csv".format(dataName))
        test_df = pd.read_csv(DIR + "{}-test.csv".format(dataName))

        if self.train:
            tmp_df= train_df.append(val_df)
            targets = tmp_df.review_score.values
            tmp_df.drop(['review_id', 'order_id', 'review_score', 'product_id'], axis=1, inplace=True)
            data = tmp_df.values.astype(np.float32)
        else:
            targets = test_df.review_score.values
            test_df.drop(['review_id', 'order_id', 'review_score', 'product_id'], axis=1, inplace=True)
            data = test_df.values.astype(np.float32)

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode="L")

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}


    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"





def Brazil(data_path, permuted=False, permutation_seed=None):
    channel = 1
    # im_size = (54)
    num_classes = 5

    data_path = '/home/anonymous/disk/gits/FastCore/fastcore/dataFile'
    print('see data path is ', data_path)

    dst_train = brazil(data_path, train=True, download=False, transform=None)
    dst_test  = brazil(data_path, train=False, download=False, transform=None)

    im_size = dst_train.data.shape[1]
    class_names = [str(c) for c in range(num_classes)]

    return channel, im_size, num_classes, class_names, -1, -1, dst_train, dst_test
