import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder
import numpy as np

''' Logistic Regression '''


class LogisticRegression(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(LogisticRegression, self).__init__()
        """ only a single linear layer, which already contains bias"""
        if np.isscalar(im_size):
            input_size = im_size * channel
        else:
            input_size = im_size[0] * im_size[1] * channel
        self.fc_1 = nn.Linear(input_size, num_classes)
        # self.fc_2 = nn.Linear(128, 128)
        # self.fc_3 = nn.Linear(128, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        # self.embedding_recorder.embedding=
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_1

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = x.view(x.size(0), -1)
            out = self.embedding_recorder(out)
            # out = F.relu(self.fc_1(out))
            # out = F.relu(self.fc_2(out))
            # out = self.embedding_recorder(out)
            out = self.fc_1(out)
        return out
