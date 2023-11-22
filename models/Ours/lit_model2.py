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

import pytorch_lightning as pl
import torch
import torch.nn as nn
from myutils.torch.audio_df_detection import BinaryClassification
from myutils.torch.losses import (
    BinaryTokenContrastLoss,
    LabelSmoothingBCE,
    MultiClass_ContrastLoss,
)
from myutils.torch.optim import Adam_GC
from myutils.torchaudio.transforms import AddGaussianSNR

from copy import deepcopy

# + editable=true slideshow={"slide_type": ""}
from .model2 import AudioModel


# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# from model2 import AudioModel

# + editable=true slideshow={"slide_type": ""}
class AudioModel_lit(BinaryClassification):
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.model = AudioModel(
            dims=cfg.dims,
            n_blocks=cfg.n_blocks,
            vocoder_classes=cfg.method_classes,
            cfg=cfg,
        )

        self.beta1, self.beta2, self.beta3 = cfg.beta

        self.one_stem = cfg.one_stem
        
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.contrast_lossN = MultiClass_ContrastLoss(alpha=2.5, distance="l2")
        # self.contrast_lossN = MultiClass_ContrastLoss(alpha=0.1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.save_hyperparameters()

    def calcuate_loss_one_stem(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]

        batch_res["logit"] = batch_res["content_logit"]
        label = batch["label"]
        label_32 = label.type(torch.float32)
        losses = {}

        losses["cls_loss"] = self.bce_loss(
            batch_res["logit"].squeeze(), label_32
        )
        losses["content_contrast_loss"] = self.contrast_loss2(
            batch_res["content_feature"], label_32
        )

        self.log_dict(
            {f"{stage}-{key}-step": losses[key] for key in losses},
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True,
        )
        self.log_dict(
            {f"{stage}-{key}": losses[key] for key in losses},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        # loss = sum(losses.values()) / (len(losses))
        loss = (losses["cls_loss"] + losses["content_contrast_loss"]) / 2
        return loss

    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]

        label = batch["label"]
        label_32 = label.type(torch.float32)
        losses = {}

        losses["cls_loss"] = self.bce_loss(batch_res["logit"].squeeze(), label_32)
        losses["cls_loss_aug"] = self.bce_loss(
            batch_res["aug_logit"].squeeze(), label_32
        )

        losses["voc_loss"] = self.cross_entropy_loss(
            batch_res["vocoder_logit"], batch["vocoder_label"]
        )
        losses["voc_contrast_loss"] = self.contrast_lossN(
            batch_res["vocoder_feature"], batch["vocoder_label"].type(torch.float32)
        )
        losses["content_loss"] = self.bce_loss(
            batch_res["content_logit"].squeeze(), label_32
        )
        losses["content_contrast_loss"] = self.contrast_loss2(
            batch_res["content_feature"], label_32
        )
        losses["content_adv_loss"] = self.cross_entropy_loss(
            batch_res["content_voc_logit"],
            torch.ones_like(batch_res["content_voc_logit"])
            * (1 / batch_res["content_voc_logit"].shape[-1]),
        )

        self.log_dict(
            {f"{stage}-{key}-step": losses[key] for key in losses},
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True,
        )
        self.log_dict(
            {f"{stage}-{key}": losses[key] for key in losses},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        # loss = sum(losses.values()) / (len(losses))
        loss = (
            losses["cls_loss"]
            + self.beta1 * losses["cls_loss_aug"]
            + self.beta3
            * (
                losses["content_adv_loss"]
                + losses["content_loss"]
                + losses["content_contrast_loss"]
            )
            + self.beta2 * (losses["voc_loss"] + losses["voc_contrast_loss"])
        ) / 7
        return loss

    def configure_optimizers(self):
        optim = Adam_GC  # or torch.optim.Adam
        # optim = torch.optim.Adam
        optimizer = optim(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        batch_res = self.model(audio, stage=stage)
        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()
        return batch_res

    def _shared_eval_step(self, batch, batch_idx, stage="train"):
        batch_res = self._shared_pred(batch, batch_idx, stage=stage)

        label = batch["label"]
        if not self.one_stem:
            loss = self.calcuate_loss(batch_res, batch, stage=stage)
        else:
            loss = self.calcuate_loss_one_stem(batch_res, batch, stage=stage)
        
        self.log_dict(
            {f"{stage}-loss": loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        batch_res["loss"] = loss
        # print(batch_res['pred'], batch_res['logit'], label)
        return batch_res
