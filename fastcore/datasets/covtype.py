from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils.data.dataset import *
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import check_integrity


def processCovtype(data_path, save_path):

    num, dim = 581012, 54

    X = np.zeros((num, dim), dtype=np.float32)
    y = np.zeros(num, dtype=np.int32)
    path = data_path

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            y[i] = float(line.split()[0])
            for e in line.split()[1:]:
                cur = e.split(':')
                X[i][int(cur[0]) - 1] = float(cur[1])
            i += 1
    y = np.array(y, dtype=np.int32)
    y = y - np.ones(len(y), dtype=np.int32)

    N = len(X)
    NUM_TRAINING, NUM_VALIDATION = int(N / 2), int(N / 2) + int(N / 4)

    sample = np.arange(N)
    np.random.seed(0)
    np.random.shuffle(sample)
    train_sample, val_sample, test_sample = \
        sample[:NUM_TRAINING], sample[NUM_TRAINING:NUM_VALIDATION], sample[NUM_VALIDATION:]

    X_train, y_train = X[train_sample, :], y[train_sample]
    X_val, y_val = X[val_sample, :], y[val_sample]
    X_test, y_test = X[test_sample, :], y[test_sample]

    np.save(os.path.join(save_path, "train-X.npy"), X_train)
    np.save(os.path.join(save_path, "train-Y.npy"), y_train)

    np.save(os.path.join(save_path, "test-X.npy"), X_test)
    np.save(os.path.join(save_path, "test-Y.npy"), y_test)
    return


class covtype(Dataset):
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


    resources = [
        "train-X.npy",
        "train-Y.npy",
        "test-X.npy",
        "test-Y.npy"
    ]

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
        self.classes = [0, 1]
        if not self._check_exists():
            if os.path.isfile(os.path.join(self.raw_folder, "covtype.libsvm.binary.scale")):
                """ Preprocess the data from scratch """
                data_dir = os.path.join(self.raw_folder, "covtype.libsvm.binary.scale")
                save_dir = self.raw_folder
                processCovtype(data_dir, save_dir)
            else:
                raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()


    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url in self.resources
        )


    def _load_data(self):


        feature_file = f"{'train' if self.train else 'test'}-X.npy"
        # data = read_image_file(os.path.join(self.raw_folder, feature_file))
        data = np.load(os.path.join(self.raw_folder,feature_file))

        label_file = f"{'train' if self.train else 'test'}-Y.npy"
        # targets = read_label_file(os.path.join(self.raw_folder, label_file))
        targets = np.load(os.path.join(self.raw_folder, label_file))

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

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

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





def Covtype(data_path, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (54)
    num_classes = 2

    mean = [0.1307]
    std = [0.3081]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])


    data_path = '/home/anonymous/disk/gits/FastCore/fastcore/dataFile'
    print('see data path is ', data_path)

    dst_train = covtype(data_path, train=True, download=False, transform=transform)
    dst_test  = covtype(data_path, train=False, download=False, transform=transform)

    class_names = [str(c) for c in range(num_classes)]


    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
