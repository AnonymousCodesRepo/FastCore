from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data.dataset import *
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import check_integrity
import pandas as pd



class imdb(Dataset):
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

        # dataName = "IMDBC5"
        dataName = "IMDBC5"
        trainX = np.load('/home/anonymous/disk/C-craig/dataset/{}-train-X.npy'.format(dataName))
        valX = np.load('/home/anonymous/disk/C-craig/dataset/{}-val-X.npy'.format(dataName))
        testX = np.load('/home/anonymous/disk/C-craig/dataset/{}-test-X.npy'.format(dataName)).astype(np.float32)
        trainX = np.concatenate((trainX, valX)).astype(np.float32)

        trainY = np.load('/home/anonymous/disk/C-craig/dataset/{}-train-y.npy'.format(dataName))
        valY = np.load('/home/anonymous/disk/C-craig/dataset/{}-val-y.npy'.format(dataName))
        testY = np.load('/home/anonymous/disk/C-craig/dataset/{}-test-y.npy'.format(dataName)).astype(np.float32)
        trainY = np.concatenate((trainY, valY)).astype(np.float32)


        idxs = np.argwhere(trainY != -1).reshape(-1)
        trainX = trainX[idxs,:-1]
        trainY = trainY[idxs]

        idxs = np.argwhere(testY != -1).reshape(-1)
        testX = testX[idxs,:-1]
        testY = testY[idxs]


        if self.train:
            data = trainX
            targets = trainY
        else:
            data = testX
            targets = testY
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





def IMDB(data_path, permuted=False, permutation_seed=None):
    channel = 1
    # im_size = (54)
    num_classes = 5


    data_path = '/home/anonymous/disk/gits/FastCore/fastcore/dataFile'
    print('see data path is ', data_path)

    dst_train = imdb(data_path, train=True, download=False, transform=None)
    dst_test  = imdb(data_path, train=False, download=False, transform=None)

    im_size = dst_train.data.shape[1]
    class_names = [str(c) for c in range(num_classes)]


    return channel, im_size, num_classes, class_names, -1, -1, dst_train, dst_test
