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

# + editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# +
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
# -

from .datasets import ADD2023, WaveFake
from .tools import AudioDataset

from myutils.torch.transforms.audio import AudioRawBoost, SpecAugmentTransform_Wave


# # Different Datasets

# ## WaveFake

def make_WaveFake(cfg):
    dataset = WaveFake(root_path=cfg.root_path)
    data = dataset.get_sub_data(cfg.subsets, splits=cfg.splits)
    return data


# ## ADD 2023

# + editable=true slideshow={"slide_type": ""}
def make_ADD2023(cfg):
    add2023 = ADD2023(root_path=cfg.root_path)

    res = {}

    res["test1"] = add2023.read_test_metedata(track="1.2", test_round=1)
    res["test2"] = add2023.read_test_metedata(track="1.2", test_round=2)
    res["train"] = add2023.read_train_dev_metadata(track="1.2", train_or_dev=["train", 'dev'])
    res['vc'] =  add2023.read_voice_conversion_metadata(track="1.2")
    res['train'] = pd.concat([res['train'], res['vc']], ignore_index=True)
    res["dev"] = add2023.read_train_dev_metadata(track="1.2", train_or_dev="dev")

    return res


# -

# # Make dataset

# + editable=true slideshow={"slide_type": ""}
def make_trainset(data):
    transforms = Compose([AudioRawBoost(), SpecAugmentTransform_Wave()])
    # transforms = AudioRawBoost()
    # transforms = None
    
    _ds = AudioDataset(
        data,
        len_clip=48000,
        len_sep=32000,
        audio_split=False,
        transform=transforms,
        over_sample=True,
        random_cut=True,
    )
    return _ds


# -

def make_val_or_test_set(data):
    _ds = AudioDataset(
        data,
        len_clip=48000,
        len_sep=32000,
        audio_split=False,
        transform=None,
        over_sample=False,
        random_cut=False,
    )
    return _ds


def make_dl(cfg):
    if cfg.name == "WaveFake":
        datas = make_WaveFake(cfg.WaveFake)
    elif cfg.name == "ADD2023":
        datas = make_ADD2023(cfg.ADD2023)

    datasets = {}
    for key in datas.keys():
        if key == "train":
            datasets["train"] = make_trainset(datas["train"])
        else:
            datasets[key] = make_val_or_test_set(datas[key])

    if cfg.to_dl:
        dataloaders = {}
        for key in datasets.keys():
            dataloaders[key] = DataLoader(
                datasets[key],
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=True if key == "train" else False,
                shuffle=True if key == "train" else False,
                prefetch_factor=2,
                collate_fn=None,
            )
        return dataloaders
    else:
        return datasets

# + tags=["style-student", "active-ipynb"]
# import sys
#
# sys.path.append("/home/ay/zky/Coding/0-Audio")
# from config import get_cfg_defaults
#
# cfg = get_cfg_defaults()
# train, val, test = make_WaveFake(cfg.DATASET.WaveFake)
# ds = AudioDataset(train, cut=500000)
# ds = make_dl(cfg.DATASET)
