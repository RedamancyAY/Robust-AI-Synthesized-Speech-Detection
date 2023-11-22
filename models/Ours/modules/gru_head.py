import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils.torch.nn import LambdaFunctionModule


class GRU_Head(nn.Module):
    def __init__(
        self,
        n_dim,
        gru_node=512,
        gru_layers=3,
    ):
        super().__init__()

        self.gru_norm = nn.Sequential(
            nn.BatchNorm1d(n_dim), nn.LeakyReLU(negative_slope=0.3)
        )
        self.gru = nn.GRU(
            input_size=n_dim,
            hidden_size=gru_node,
            num_layers=gru_layers,
            batch_first=True,
        )

    def forward(self, x):
        '''
            (batch, filt, time) >> (batch, time, filt)
        '''
        x = self.gru_norm(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return x
