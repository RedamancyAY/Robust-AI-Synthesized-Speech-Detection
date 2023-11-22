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

# ## Import

# +
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from myutils.torch.nn import LambdaFunctionModule
from myutils.torchaudio.transforms import SpecAugmentBatchTransform

# + editable=true slideshow={"slide_type": ""}
from .gru_head import GRU_Head
from .model_RawNet2 import LayerNorm, SincConv_fast
from .msfm import MultiScaleFusion, MultiScaleFusion2D
from .utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from gru_head import GRU_Head
# from model_RawNet2 import LayerNorm, SincConv_fast
# from msfm import MultiScaleFusion, MultiScaleFusion2D
# from utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention
# -

# ## Build stage

def build_stage(
    n_dim_in,
    n_dim_out,
    n_blocks,
    samples_per_frame,
    n_head=1,
    downsample_factor=1,
    use2D=False,
):
    module = MultiScaleFusion if not use2D else MultiScaleFusion2D
    # print(n_dim_in, n_dim_out)
    conv1 = nn.Conv1d(n_dim_in, n_dim_out, 3, stride=1, padding=1)
    conv_blocks = [
        module(
            n_dim=n_dim_out,
            n_head=n_head,
            samples_per_frame=samples_per_frame,
        )
        for i in range(n_blocks)
    ]
    module = nn.Sequential(conv1, *conv_blocks)
    if downsample_factor > 1:
        module.add_module(
            "down-sample", nn.Conv1d(n_dim_out, n_dim_out, 5, stride=2, padding=2)
        )
    return module


def build_stage2D(
    n_dim_in, n_dim_out, n_blocks, samples_per_frame, n_head=1, downsample_factor=1
):
    # print(n_dim_in, n_dim_out)
    conv1 = nn.Conv2d(n_dim_in, n_dim_out, 3, stride=1, padding=1, bias=False)
    conv_blocks = [
        MultiScaleFusion2D(
            n_dim=n_dim_out,
            n_head=n_head,
            samples_per_frame=samples_per_frame,
        )
        for i in range(n_blocks)
    ]
    module = nn.Sequential(conv1, *conv_blocks)
    if downsample_factor > 1:
        module.add_module(
            "down-sample",
            # nn.Conv2d(n_dim_out, n_dim_out, 3, stride=2, padding=2)
            nn.Sequential(
                nn.Conv2d(n_dim_out, n_dim_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(n_dim_out),
            ),
        )
    return module


# ## FeatureExtractor

# + jupyter={"source_hidden": true}
class FeatureExtractor(nn.Module):
    def __init__(
        self,
        dims=[32, 32, 64, 64, 128],
        n_blocks=[2, 2, 2, 4, 2],
        n_heads=[1, 1, 2, 2, 4],
        samples_per_frame=400,
        use_gru_head=False,
        gru_node=512,
        gru_layers=3,
    ):
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353)

        self.samples_per_frame = samples_per_frame
        self.conv_head = nn.Sequential(
            nn.Conv1d(1, dims[0], 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(dims[0], dims[0], 3, stride=1, padding=1),
        )
        # self.conv_head = nn.Sequential(
        #     SincConv_fast(out_channels=dims[0], kernel_size=1024, padding=512),
        #     LambdaFunctionModule(lambda x: torch.abs(x)),
        #     nn.MaxPool1d(4),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(negative_slope=0.3),
        #     nn.Conv1d(dims[0], dims[0], 3, stride=1, padding=1),
        # )
        self.stages = nn.ModuleList(
            [
                build_stage(
                    n_dim_in=dims[max(i - 1, 0)],
                    n_dim_out=dims[i],
                    n_blocks=n_blocks[i],
                    n_head=n_heads[i],
                    samples_per_frame=samples_per_frame // (4 * (2**i)),
                    downsample_factor=2 if i < len(dims) - 1 else 1,
                )
                for i in range(len(dims))
            ]
        )

        self.use_gru_head = use_gru_head
        if use_gru_head:
            self.gru_head = GRU_Head(
                n_dim=dims[-1], gru_node=gru_node, gru_layers=gru_layers
            )

    def get_feature(self, x):
        audio_length = x.shape[-1]
        audio_frames = audio_length // self.samples_per_frame

        x = self.conv_head(x)
        for i, stage in enumerate(self.stages):
            # print("Input of the %d-th stage"%(i+1), x.shape)
            x = stage(x)  # (B, C, frames)
            # print("Output of the %d-th stage"%(i+1), x.shape)
        return x

    def forward(self, x):
        x = (x - torch.mean(x, dim=(1, 2), keepdim=True)) / (
            torch.std(x, dim=(1, 2), keepdim=True) + 1e-9
        )
        x = self.get_feature(x)  # (B, C, T)
        if self.use_gru_head:
            x = self.gru_head(x)
        return x


# -

# ### 2D

# + editable=true slideshow={"slide_type": ""}
class FeatureExtractor2D(nn.Module):
    def __init__(
        self,
        dims=[32, 64, 64, 128],
        n_blocks=[1, 1, 2, 1],
        n_heads=[1, 2, 2, 4],
        samples_per_frame=400,
        use_gru_head=False,
        gru_node=512,
        gru_layers=3,
    ):
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        self.dims = dims
        self.samples_per_frame = samples_per_frame
        self.conv_head = nn.Sequential(
            nn.Conv2d(1, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], 3, stride=1, padding=1, bias=False),
        )

        # print(dims)
        self.stages = nn.ModuleList(
            [
                build_stage2D(
                    n_dim_in=dims[max(i - 1, 0)],
                    n_dim_out=dims[i],
                    n_blocks=n_blocks[i],
                    n_head=n_heads[i],
                    samples_per_frame=samples_per_frame // (4 * (2**i)),
                    downsample_factor=2 if i < len(dims) - 1 else 1,
                    # downsample_factor=2,
                )
                for i in range(len(dims))
            ]
        )

        # self.conv_tail = nn.Sequential(
        #     nn.Conv2d(dims[-1], dims[-1], 3, stride=1, padding=1),
        #     # nn.BatchNorm2d(dims[-1]),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.transform = SpecAugmentBatchTransform.from_policy("ld")

    def preprocess(self, x, stage='train'):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)

        if stage=='train':
            x = self.transform(x)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        return x

    def pool_reshape(self, x):
        # print(x.shape)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return x

    def get_feature(self, x):
        x = self.conv_head(x)
        for i, stage in enumerate(self.stages):
            # print("Input of the %d-th stage"%(i+1), x.shape)
            x = stage(x)  # (B, C, frames)
            # print("Output of the %d-th stage" % (i + 1), x.shape)
        return x

    def get_hidden_state(self, x, stage_id, stage='train'):
        x = self.preprocess(x,stage=stage)
        x = self.conv_head(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # print(i, x.shape)
            if i == stage_id:
                break
        return x

    def get_final_feature(self, x, stage_id):
        for i, stage in enumerate(self.stages):
            if i <= stage_id:
                continue
            x = stage(x)
            # print(i, x.shape)

        x = self.pool_reshape(x)
        return x

    def forward(self, x):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )

        x = self.get_feature(x)  # (B, C, T)

        # x = self.conv_tail(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return x
