# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
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
from .gradient_reversal import GradientReversal
from .modules.classifier import Classifier
from .modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
from .modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast
from .utils import weight_init


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from gradient_reversal import GradientReversal
# from modules.classifier import Classifier
# from modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
# from modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast
# from utils import weight_init

# + editable=true slideshow={"slide_type": ""}
class AudioModel(nn.Module):
    def __init__(
        self,
        dims=[32, 64, 64, 128],
        n_blocks=[1, 1, 2, 1],
        n_heads=[1, 2, 2, 4, 1, 1],
        samples_per_frame=640,
        gru_node=128,
        gru_layers=3,
        fc_node=128,
        num_classes=1,
        vocoder_classes=8,
        adv_vocoder=False,
        cfg=None
    ):
        super().__init__()

        self.cfg=cfg
        
        # self.norm = LayerNorm(48000)
        self.feature_model = FeatureExtractor2D(
            dims=dims,
            n_blocks=n_blocks,
            n_heads=n_heads,
            samples_per_frame=samples_per_frame,
            use_gru_head=False,
            gru_node=gru_node,
            gru_layers=gru_layers,
        )
        # self.feature_model = RawNet_FeatureExtractor()

        self.classifier_c = Classifier(
            dims=dims,
            fc_node=dims[-1],
            num_classes=1,
            voc_head=False,
            use_gru_head=False,
            gru_node=gru_node,
            gru_layers=gru_layers,
        )
        self.adv_vocoder = adv_vocoder
        if adv_vocoder:
            self.classifier_v = Classifier(
                dims=dims,
                fc_node=fc_node,
                num_classes=vocoder_classes,
                use_gru_head=True,
                gru_node=gru_node,
                gru_layers=gru_layers,
            )
            self.grl = GradientReversal(alpha=1.0)

        self.apply(weight_init)

    def forward(self, x, stage="test"):
        res = {}
        # x = self.norm(x)


        if self.cfg.nograd:
            with torch.no_grad():
                x = self.feature_model(x)  # (B, C, T)
        else:
            x = self.feature_model(x)
        # print(x.shape)

        res["content_feature"], res["logit"] = self.classifier_c(x)
        if self.adv_vocoder:
            res["vocoder_feature"], res["vocoder_logit"] = self.classifier_v(
                self.grl(x)
            )
        return res

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = AudioModel()
# x = torch.randn(32, 1, 48000)
# model(x)
# with torch.autograd.profiler.profile(enabled=True) as prof:
#     x = torch.randn(2, 1, 48000)
#     _ = model(x).shape
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model.to("cuda:0")
#
# import torch
# from torch.autograd import Variable
#
# x = torch.randn(16, 1, 48000)
# y = Variable(x, requires_grad=True).to("cuda:0")

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     z = model(y)
#     print(y.shape)
#     z = torch.sum(z["logit"])
#     z.backward()
# # NOTE: some columns were removed for brevityM
# print(prof.key_averages().table(sort_by="self_cuda_time_total"))
