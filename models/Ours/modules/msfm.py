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
from myutils.torch.nn import LambdaFunctionModule

# + editable=true slideshow={"slide_type": ""}
from .conv_attention import MLP, Attention
from .utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from conv_attention import MLP, Attention
# from utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention
# -

#
# ## Multi-Scale Fusion Module

# + editable=true slideshow={"slide_type": ""}
class MultiScaleFusion(nn.Module):
    def __init__(self, n_dim, n_head=1, scales=[1, 5, 10], samples_per_frame=400):
        super().__init__()

        self.n_dim = n_dim
        self.samples_per_frame = samples_per_frame
        self.norm = nn.BatchNorm1d(n_dim)

        scales = [1, 5, 10]
        assert samples_per_frame % scales[-1] == 0, samples_per_frame

        self.down_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool1d(scales[i] * 3, stride=scales[i], padding=scales[i])
                    if i > 0
                    else nn.Identity(),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                    # nn.LeakyReLU(negative_slope=0.3),
                    # nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                )
                for i in range(3)
            ]
        )

        self.up_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=scales[i]) if i > 0 else nn.Identity(),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                    # nn.LeakyReLU(negative_slope=0.3),
                    # nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                )
                for i in range(3)
            ]
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv1d(n_dim * 3, n_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(n_dim, n_dim * 3, 3, stride=1, padding=1),
        )
        self.mha = Multi_Head_Attention(
            max_k=80, embed_dim=n_dim, num_heads=n_head, dropout=0.1
        )
        self.attn_upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=samples_per_frame // scales[i]),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                    # nn.LeakyReLU(negative_slope=0.3),
                    # nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                )
                for i in range(3)
            ]
        )

        self.register_parameter("alpha", nn.Parameter(torch.ones(1, n_dim, 1)))
        self.register_parameter("beta", nn.Parameter(torch.ones(1, n_dim * 3, 1)))

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        n_frames = x.shape[-1] // self.samples_per_frame
        avg_pool = partial(F.adaptive_avg_pool1d, output_size=n_frames)
        max_pool = partial(F.adaptive_max_pool1d, output_size=n_frames)

        frame_feat = []
        ms_feat = []
        for i in range(3):
            y = self.down_samples[i](x)
            print("scale %d : " % i, y.shape)
            ms_feat.append(y)
            attn = avg_pool(y) + max_pool(y)  # (B, n_dim, n_frames)
            frame_feat.append(attn)
            # frame_feat.append(attn.transpose(1, 2))  # (B, n_frames, n_dim)

        frame_feat = torch.concat(frame_feat, dim=1)  # (B, 3*n_dim, n_frames)
        frame_feat = self.conv_fusion(frame_feat)
        frame_feat = torch.split(frame_feat, self.n_dim, dim=1)
        frame_feat = [x.transpose(1, 2) for x in frame_feat]

        v, k, q = frame_feat
        attn = self.mha(q, k, v)
        attn = attn.transpose(1, 2)  # (B, n_dim, n_frames)
        # print("attn shape: ", attn.shape)

        for i in range(3):
            _attn = self.attn_upsamples[i](attn)
            ms_feat[i] = ms_feat[i] * _attn
            # ms_feat[i] = (
            #     ms_feat[i]
            #     + self.beta[:, i * self.n_dim : (i + 1) * self.n_dim, :] * _attn
            # )

        rec_feat = []
        for i in range(3):
            y = self.up_samples[i](ms_feat[i])
            rec_feat.append(y)

        rec_feat = rec_feat[0] + rec_feat[1] + rec_feat[2]
        x = x + self.alpha * rec_feat
        return x


# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = MultiScaleFusion(n_dim=32)
# x = torch.randn(2, 32, 4000)
# model(x)
# -

# ## 2D

# + editable=true slideshow={"slide_type": ""}
class MultiScaleFusion2D(nn.Module):
    def __init__(self, n_dim, n_head=1, scales=[1, 5, 10], samples_per_frame=400):
        super().__init__()

        self.n_dim = n_dim
        self.norm = nn.BatchNorm2d(n_dim)

        scales = [1, 2, 3]

        self.down_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool2d(scales[i] * 3, stride=scales[i], padding=scales[i])
                    if i > 0
                    else nn.Identity(),
                    nn.Conv2d(
                        n_dim, n_dim, 3, stride=1, padding=1, groups=1, bias=False
                    ),
                )
                for i in range(3)
            ]
        )

        # self.conv_attention = Attention(dim=n_dim)
        self.conv_attention = nn.Sequential(
            Attention(dim=n_dim), MLP(dim=n_dim, mlp_ratio=2.0)
        )

        self.up_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=scales[i], mode="bilinear", align_corners=True
                    )
                    if i > 0
                    else nn.Identity(),
                    nn.Conv2d(
                        n_dim, n_dim, 3, stride=1, padding=1, groups=1, bias=False
                    ),
                )
                for i in range(3)
            ]
        )

        # self.final_proj = nn.Sequential(
        #     nn.Conv2d(n_dim*3, n_dim, 1, bias=False),
        #     nn.BatchNorm2d(n_dim),
        #     nn.Dropout(0.1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(n_dim, n_dim, 1, bias=False)
        # )
        # self.final_proj = nn.Conv2d(n_dim*3, n_dim, 1, bias=False)

        self.register_parameter("alpha1", nn.Parameter(torch.ones(1, n_dim, 1, 1)))
        self.register_parameter("alpha2", nn.Parameter(torch.ones(1, n_dim, 1, 1)))
        self.register_parameter("alpha3", nn.Parameter(torch.ones(1, n_dim, 1, 1)))
        self.register_parameter("alpha", nn.Parameter(torch.ones(1, n_dim, 1, 1)))

    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        x = self.norm(x)

        frame_feat = []
        ms_feat = []
        for i in range(3):
            y = self.down_samples[i](x)
            y = self.conv_attention(y)
            # print("scale %d : " % i, y.shape)
            ms_feat.append(y)

        rec_feat = []
        for i in range(3):
            y = self.up_samples[i](ms_feat[i])
            _H, _W = y.shape[-2], y.shape[-1]
            y = F.pad(y, (0, W - _W, 0, H - _H))
            # print(y.shape)
            rec_feat.append(y)

        # rec_feat = (rec_feat[0] + rec_feat[1] + rec_feat[2]) / 3
        rec_feat = (
            self.alpha1 * rec_feat[0]
            + self.alpha2 * rec_feat[1]
            + self.alpha3 * rec_feat[2]
        ) / 3
        # rec_feat = self.final_proj(torch.concat(rec_feat, dim=1))
        x = x + self.alpha * rec_feat
        # x = x + rec_feat
        return x

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# module = MultiScaleFusion2D(n_dim=64)
# x = torch.randn(2, 64, 224, 252)
# module(x).shape

# + editable=true slideshow={"slide_type": ""}
# spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
# x = torch.randn(2, 1, 48000)
# spectrogram(x).shape
