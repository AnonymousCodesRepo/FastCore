import numpy as np
# from .coresetmethod import CoresetMethod
# from fastcore.group import *
from .earlytrain import EarlyTrain
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
from fastcore.methods.scale import Scale


import torch
from .methods_utils import FacilityLocation, submodular_optimizer
from ..nets.nets_utils import MyDataParallel
from typing import Any, Callable, Dict, List, Optional, Tuple



from torch.utils.data import Dataset
import warnings


class TensorDataset(Dataset):

    def __init__(self, data_tensor, target_tensor):
        # self.data_tensor = data_tensor
        # self.target_tensor = target_tensor

        """ TODO """
        self.classes = np.unique(target_tensor)
        self.transform = None
        self.target_transform = None
        self.data, self.targets = data_tensor, target_tensor

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

    def __getitem__(self, index):
        data, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}




class ScaleDNN(EarlyTrain):

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, balance=False, specific_model=None,
                 groupNum=1000, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)
        self.balance = balance
        self.n_train = len(dst_train)
        self.groupNum = groupNum
        self.args = args


    # def calc_gradient(self):
    #     pass
    #
    # def finish_run(self):
    #     pass
    #
    # def select(self):
    #     pass


    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def calc_gradient(self, index=None):
        self.model.eval()

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch, num_workers=self.args.workers)
        sample_num = len(self.dst_val.targets) if index is None else len(index)
        self.embedding_dim = self.model.get_last_layer().in_features

        gradients = []

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(torch.nn.functional.softmax(outputs.requires_grad_(True), dim=1),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1,
                                                                                       self.embedding_dim).repeat(1,
                                                                                                                  self.args.num_classes,
                                                                                                                  1) * bias_parameters_grads.view(
                    batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients.append(
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy())

        # (n, dim_gradient)
        gradients = np.concatenate(gradients, axis=0)
        return gradients
        # self.model.train()
        # return euclidean_dist_pair_np(gradients)

    def calc_weights(self, matrix, result):
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

      
    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        self.model.no_grad = True
        with self.model.embedding_recorder:
            if self.balance:
                # Do selection by class
                selection_result = np.array([], dtype=np.int32)
                weights = np.array([])

                all_gradient = None
                for c in range(self.args.num_classes):
                    class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                    c_gradient  = self.calc_gradient(class_index)

                    if all_gradient is None:
                        all_gradient = np.empty((self.n_train, c_gradient.shape[1]))
                    all_gradient[class_index,:] = c_gradient


            # dst_train = covtype(data_path, train=True, download=False, transform=transform)

            print(f"#### 【Gradient dim is {all_gradient.shape[1]}】")
            dst_train = TensorDataset(all_gradient, self.dst_train.targets)
            print("all_gradient")
            print(all_gradient)
            print()
            # scale =
        print()
        scale = Scale(dst_train, self.args, self.args.fraction, self.args.seed)
        self.model.no_grad = False
        return scale.select()
        # return {"indices": selection_result, "weights": weights}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
