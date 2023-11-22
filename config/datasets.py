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

from yacs.config import CfgNode as ConfigurationNode

from argparse import Namespace
from typing import Any, NamedTuple


# # 默认配置

def NameCfgNode(**kwargs):
    x = ConfigurationNode(kwargs)
    return x



ROOT_PATHs = NameCfgNode(
    WaveFake="xxxx",
    LibriSeVoc="xxx",
    DECRO="xxxx",
)

# ### WaveFake

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# class NameCfgNode2(ConfigurationNode):
#     def __init__(self, **kwargs):
#         # x = Namespace(**kwargs)
#         # print(x, vars(x))
#         print(kwargs)
#         super().__init__(kwargs)
# -

WaveTasks = {
    "inner_eval": NameCfgNode(
        trainset=0, methods=[0, 1, 2, 3, 4, 5, 6], splits=[64_000, 16_000, 24_800]
    ),
    "cross_lang": NameCfgNode(
        train=NameCfgNode(trainset=0, methods=[1, 2], splits=[0.8, 0.2]),
        test=NameCfgNode(trainset=1, methods=[1, 2], splits=[1.0]),
    ),
    "cross_method": NameCfgNode(
        train=NameCfgNode(trainset=0, methods=[0, 5], splits=[0.8, 0.2]),
        test=[
            NameCfgNode(trainset=0, methods=[i], splits=[1.0]) for i in [1, 2, 3, 4, 6]
        ],
    ),
    "cross_method2": NameCfgNode(
        train=NameCfgNode(trainset=0, methods=[0, 1], splits=[0.8, 0.2]),
        test=[
            NameCfgNode(trainset=0, methods=[i], splits=[1.0]) for i in [2, 3, 4, 5, 6]
        ],
    ),
}


def WaveFake(task="inner_eval"):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs['WaveFake']
    C.task = task
    if task == "inner_eval":
        task = WaveTasks[task]
        C.trainset = task.trainset  # 0 / 1
        C.methods = task.methods  # 0-6
        C.splits = task.splits
    else:
        try:
            C.task_cfg = WaveTasks[task]
        except KeyError:
            return C
    return C


# ### LibriSeVoc

LibriSeVocTasks = {
    "inner_eval": NameCfgNode(
        methods=[0, 1, 2, 3, 4, 5], splits=[55_440, 18_480, 18_487]
    ),
    "cross_method": NameCfgNode(
        train=NameCfgNode(methods=[0, 4], splits=[0.8, 0.2]),
        test=[NameCfgNode(methods=[i], splits=[1.0]) for i in [1, 2, 3, 5]],
    ),
    "cross_dataset": NameCfgNode(
        train=NameCfgNode(methods=[0, 1, 2, 3, 4, 5], splits=[0.8, 0.2]),
        test=[
            NameCfgNode(dataset="WaveFake", trainset=0, methods=[i], splits=[1.0])
            for i in [0, 1, 2, 3, 4, 5, 6]
        ],
    ),
}


def LibriSeVoc(task="inner_eval"):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.task = task
    if task == "inner_eval":
        task = LibriSeVocTasks[task]
        C.methods = task.methods  # 0-5
        C.splits = task.splits
    else:
        try:
            C.task_cfg = LibriSeVocTasks[task]
        except KeyError:
            return C
    return C


# ### DECRO 

def DECRO(task=None):
    C = ConfigurationNode()
    C.ROOT_PATHs = ROOT_PATHs
    C.root_path = ROOT_PATHs.DECRO
    C.task = task
    C.main = "en" if task == "en->ch" else "ch"
    return C





# # Settings

def transforms():
    __C = ConfigurationNode()
    return __C


def train_settings():
    __C = ConfigurationNode()
    __C.sample_rate = 16000  # audio sampling ratio
    __C.max_wave_length = 48000  # audio length for training
    __C.audio_feature = None  # features like MFCC, LFCC...
    __C.max_feature_frames = 320  # max frames of feature

    __C.over_sample = True  # over sample when the labels are not balanced
    __C.localization_task = 0
    __C.batch_size = 16  # batch size
    __C.num_workers = 10  # number of worker to load dataloaders
    return __C


def test_settings():
    __C = ConfigurationNode()
    __C.sample_rate = 16000  # audio sampling ratio
    __C.max_wave_length = 48000  # audio length for training
    __C.audio_feature = None  # features like MFCC, LFCC...
    __C.max_feature_frames = 320  # max frames of feature

    __C.over_sample = False  # over sample when the labels are not balanced
    __C.localization_task = 0
    __C.batch_size = 16  # batch size
    __C.num_workers = 10  # number of worker to load dataloaders
    return __C


def get_dataset_cfg(name, task, __C=None):
    if __C is None:
        __C = ConfigurationNode()

    __C.WaveFake = WaveFake(task)
    __C.ADD2023 = ADD2023(task)
    __C.LAVDF = LAVDF(task)
    __C.DECRO = DECRO(task)
    __C.LibriSeVoc = LibriSeVoc(task)

    __C.train_settings = train_settings()
    __C.test_settings = test_settings()
    __C.transforms = transforms()
    return __C


