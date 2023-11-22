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
import os
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import Compose
# -

from myutils.torch.transforms.audio import AudioRawBoost, SpecAugmentTransform_Wave
from myutils.torchaudio.transforms import LFCC, RandomNoise, RawBoost

from .datasets import ADD2023, LAV_DF_Audio, LibriSeVoc, WaveFake, DECRO
from .tools import AudioDataset, FeatureDataset, Localization_DS, WaveDataset


# + tags=["style-activity", "active-ipynb"]
# from datasets import ADD2023, LAV_DF_Audio, LibriSeVoc, WaveFake, DECRO
# from tools import AudioDataset, FeatureDataset, Localization_DS, WaveDataset
# -

# ## ALL datasets

def make_WaveFake(cfg):
    cfg = cfg.WaveFake
    dataset = WaveFake(root_path=cfg.root_path)

    if cfg.task == "inner_eval":
        data = dataset.get_sub_data(trainset=cfg.trainset, methods=cfg.methods)
        print(data.groupby("label").count())
        data_splits = dataset.split_data(data, splits=cfg.splits)
    elif cfg.task == "cross_lang":
        task_cfg = cfg.task_cfg
        data_train = dataset.get_sub_data(
            trainset=task_cfg.train.trainset, methods=task_cfg.train.methods
        )
        train, val = dataset.split_data(
            data_train, splits=task_cfg.train.splits, return_list=True
        )
        test = dataset.get_sub_data(
            trainset=task_cfg.test.trainset, methods=task_cfg.test.methods
        )
        for _data in [train, val, test]:
            print(_data.groupby("label").count())
        data_splits = Namespace(train=train, val=val, test=test)
    elif cfg.task == "cross_method" or cfg.task == "cross_method2":
        task_cfg = cfg.task_cfg
        # get real data, and split it into train/val/test
        data_real = dataset._get_sub_data(task_cfg.train.trainset, "real")
        real_train, real_val, real_test = dataset.split_data(
            data_real, splits=[0.6, 0.2, 0.2], return_list=True
        )

        data_train = dataset.get_sub_data(
            trainset=task_cfg.train.trainset,
            methods=task_cfg.train.methods,
            contain_real=False,
        )
        train, val = dataset.split_data(
            data_train, splits=task_cfg.train.splits, return_list=True
        )
        test = [
            dataset.get_sub_data(
                trainset=_cfg.trainset, methods=_cfg.methods, contain_real=False
            )
            for _cfg in task_cfg.test
        ]
        train = pd.concat([train, real_train], ignore_index=True)
        val = pd.concat([val, real_val], ignore_index=True)
        test = [pd.concat([_test, real_test], ignore_index=True) for _test in test]
        data_splits = Namespace(train=train, val=val, test=test)

    return data_splits


def make_LibriSeVoc(cfg):
    cfg = cfg.LibriSeVoc
    dataset = LibriSeVoc(root_path=cfg.ROOT_PATHs.LibriSeVoc)
    if cfg.task == "inner_eval":
        data = dataset.get_sub_data(methods=cfg.methods)
        print(data.groupby("label").count())
        data_splits = dataset.split_data(data, splits=cfg.splits)
    elif cfg.task == "cross_method":
        task_cfg = cfg.task_cfg

        # get real data, and split it into train/val/test
        data_real = dataset.get_sub_data([], contain_real=True)
        real_train, real_val, real_test = dataset.split_data(
            data_real, splits=[0.6, 0.2, 0.2], return_list=True
        )

        data_train = dataset.get_sub_data(
            methods=task_cfg.train.methods, contain_real=False
        )
        train, val = dataset.split_data(
            data_train, splits=task_cfg.train.splits, return_list=True
        )
        test = [
            dataset.get_sub_data(methods=_cfg.methods, contain_real=False)
            for _cfg in task_cfg.test
        ]
        train = pd.concat([train, real_train], ignore_index=True)
        val = pd.concat([val, real_val], ignore_index=True)
        test = [pd.concat([_test, real_test], ignore_index=True) for _test in test]
        
        data_splits = Namespace(train=train, val=val, test=test)
    elif cfg.task == "cross_dataset":
        task_cfg = cfg.task_cfg
        data_train = dataset.get_sub_data(methods=task_cfg.train.methods)
        train, val = dataset.split_data(
            data_train, splits=task_cfg.train.splits, return_list=True
        )
        test = []
        for _cfg in task_cfg.test:
            if _cfg.dataset.lower() == "wavefake":
                dataset2 = WaveFake(root_path=cfg.ROOT_PATHs.WaveFake)
                _data = dataset2.get_sub_data(
                    trainset=_cfg.trainset, methods=_cfg.methods
                )
                test.append(_data)
        data_splits = Namespace(train=train, val=val, test=test)
    return data_splits


def make_LAVDF(cfg):
    cfg = cfg.LAVDF
    dataset = LAV_DF_Audio(root_path=cfg.root_path)
    data_splits = dataset.get_splits()
    return data_splits


def make_DECRO(cfg):
    cfg = cfg.DECRO
    dataset = DECRO(root_path=cfg.root_path)
    en_splits = dataset.get_splits(language='en')
    ch_splits = dataset.get_splits(language='ch')
    if cfg.main == 'en':
        train, val, test = en_splits.train, en_splits.val, ch_splits.test
    else:
        train, val, test = ch_splits.train, ch_splits.val, en_splits.test
    data_splits = Namespace(train=train, val=val, test=test)
    return data_splits


MAKE_DATASETS = {
    "WaveFake": make_WaveFake,
    "LAVDF": make_LAVDF,
    "LibriSeVoc": make_LibriSeVoc,
    "DECRO": make_DECRO,
}


# ## Features

def build_feature(cfg):
    if cfg.audio_feature == "LFCC":
        return LFCC()
    return None


# ## Transform

def build_transforms(cfg):

    t = RandomNoise(snr_min_db=10.0, snr_max_db=120.0, p=0.5)
    # t = RawBoost(algo=[5], p=0.5)
    return t
    return None


# ## Common Opeations

def collate_fn_for_localization_task(batch):
    res = {}
    res["fake_periods"] = [torch.tensor(x["fake_periods"]) for x in batch]
    for x in batch:
        x.pop("fake_periods")
    default_res = default_collate(batch)
    default_res.update(res)
    return default_res


def build_train_set(data: pd.DataFrame, cfg):
    ts = cfg.train_settings

    transforms = build_transforms(cfg.transforms)

    collate_fn = default_collate
    if ts.localization_task:
        _ds = Localization_DS(
            data,
            sample_rate=16_000,
            normalize=True,
            trim=False,
            max_wave_length=16_000 * 20,
            transform=torch.nn.Identity(),
            is_training=True,
        )
        collate_fn = collate_fn_for_localization_task
    # elif ts.audio_feature is not None:
    #     _ds = FeatureDataset(
    #         data,
    #         sample_rate=ts.sample_rate,
    #         trim=False,
    #         feature=build_feature(ts),
    #         max_wave_length=ts.max_wave_length,
    #         max_feature_frames=ts.max_feature_frames,
    #         transform=transforms,
    #         is_training=True,
    #     )
    else:
        _ds = WaveDataset(
            data,
            sample_rate=ts.sample_rate,
            trim=False,
            max_wave_length=ts.max_wave_length,
            transform=transforms,
            is_training=True,
        )

    _dl = DataLoader(
        _ds,
        batch_size=ts.batch_size,
        num_workers=ts.num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    return _ds, _dl


def build_val_or_test_set(data: pd.DataFrame, cfg):
    ts = cfg.test_settings
    collate_fn = default_collate

    if ts.localization_task:
        _ds = Localization_DS(
            data,
            sample_rate=16_000,
            normalize=True,
            trim=False,
            max_wave_length=16_000 * 20,
            transform=torch.nn.Identity(),
            is_training=False,
        )
        collate_fn = collate_fn_for_localization_task
    # elif ts.audio_feature is not None:
    #     _ds = FeatureDataset(
    #         data,
    #         sample_rate=ts.sample_rate,
    #         trim=False,
    #         feature=build_feature(ts),
    #         max_wave_length=ts.max_wave_length,
    #         max_feature_frames=ts.max_feature_frames,
    #         transform=None,
    #         is_training=False,
    #     )
    else:
        _ds = WaveDataset(
            data,
            sample_rate=ts.sample_rate,
            trim=False,
            max_wave_length=ts.max_wave_length,
            transform=None,
            is_training=False,
        )

    _dl = DataLoader(
        _ds,
        batch_size=ts.batch_size,
        num_workers=ts.num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    return _ds, _dl


# ## Door

def over_sample_dataset(data, column="label"):
    n_fake = len(data[data[column] == 0])
    n_real = len(data[data[column] == 1])
    if n_fake == n_real:
        return data
    if n_fake > n_real:
        sampled = data[data[column] == 1].sample(n=n_fake - n_real, replace=True)
        balanced_data = pd.concat([data, sampled])
    else:
        sampled = data[data[column] == 0].sample(n=n_real - n_fake, replace=True)
        balanced_data = pd.concat([data, sampled])

    balanced_data = balanced_data.reset_index(drop=True)
    return balanced_data


def make_data(cfg):
    
    sub_datas = MAKE_DATASETS[cfg.name](cfg)

    # print(len(sub_datas.train))
    
    data = over_sample_dataset(sub_datas.train, column="label")
    sub_datas.train = data

    train_ds, train_dl = build_train_set(sub_datas.train, cfg)
    val_ds, val_dl = build_val_or_test_set(sub_datas.val, cfg)
    if isinstance(sub_datas.test, list):
        test_ds, test_dl = [], []
        for _test in sub_datas.test:
            _ds, _dl = build_val_or_test_set(_test, cfg)
            test_ds.append(_ds)
            test_dl.append(_dl)
    else:
        test_ds, test_dl = build_val_or_test_set(sub_datas.test, cfg)

    ds = Namespace(train=train_ds, val=val_ds, test=test_ds)
    dl = Namespace(train=train_dl, val=val_dl, test=test_dl)
    return ds, dl
