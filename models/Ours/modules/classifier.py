# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils.torch.nn import LambdaFunctionModule

# + editable=true slideshow={"slide_type": ""}
from .utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention
from .gru_head import GRU_Head


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention

# + editable=true slideshow={"slide_type": ""}
class Classifier(nn.Module):
    def __init__(
        self,
        dims=[32, 32, 64, 64, 128],
        n_blocks=[2, 2, 2, 4, 2],
        n_heads=[1, 1, 2, 2, 4],
        samples_per_frame=400,
        use_gru_head=False,
        gru_node=512,
        gru_layers=3,
        num_classes=1,
        fc_node=512,
        voc_head=False,
    ):
        super().__init__()

        self.use_gru_head = use_gru_head
        if self.use_gru_head:
            self.gru_head = GRU_Head(
                n_dim=dims[-1], gru_node=gru_node, gru_layers=gru_layers
            )

        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=0.1)
        # self.fc_gru = nn.Linear(gru_node, fc_node)
        self.cls_head = nn.Linear(fc_node, num_classes)
        if voc_head:
            self.voc_head = nn.Linear(fc_node, 8)

    def forward(self, x):
        if self.use_gru_head:
            x = self.gru_head(x)

        # code = self.fc_gru(x)
        code = x
        out = self.cls_head(self.dropout(code))
        if self.num_classes == 1:
            out = out.squeeze()
        if hasattr(self, "voc_head"):
            return x, out, self.voc_head(code)
        else:
            return x, out
