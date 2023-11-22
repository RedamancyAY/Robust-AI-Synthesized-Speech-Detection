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
import os
import random
from enum import Enum
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from myutils.common.audio import get_fps_len
from myutils.tools import check_dir, read_file_paths_from_folder, to_list
from myutils.tools.pandas import DF_spliter
from pandarallel import pandarallel
from tqdm.auto import tqdm


# -

# This dataset only contains the audio tracks of the LAV-DF, which contains the following file architecture
# - train (train set of LAV-DF)
#   - xxxxxx.wav
#   - ...
# - dev (dev set of LAV-DF)
#   - xxxxxx.wav
#   - ...
# - test (test set of LAV-DF)
#   - xxxxxx.wav
#   - ...
# - metadata.json (Full metadata file)
# - metadata.min.json (Min metadata file for quick loading)
# - README.md (This file)

# | | Train | Val | Test|
# |-|-|-|-|
# |0|38178 | 15410 | 12742|
# |1|40525|16091| 13358|
# |Total| 78703| 31501|26100|

# <center><img src="https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/imgCleanShot 2023-07-07 at 10.04.46@2x.png" width="400" alt="CleanShot 2023-07-07 at 10.04.46@2x"/></center>

# |        | path                                                     | split   |   label |   n_fakes | fake_periods                    |   audio_frames |   duration |
# |-------:|:---------------------------------------------------------|:--------|--------:|----------:|:--------------------------------|---------------:|-----------:|
# |  93425 | /home/ay/data/0-原始数据集/LAV-DF-Audio/test/094691.wav  | test    |       0 |         1 | [[1.0, 1.634]]                  |         110592 |      7.04  |
# |  67199 | /home/ay/data/0-原始数据集/LAV-DF-Audio/dev/068122.wav   | val     |       0 |         1 | [[1.1, 1.404]]                  |          82944 |      5.312 |
# | 119712 | /home/ay/data/0-原始数据集/LAV-DF-Audio/dev/121264.wav   | val     |       1 |         0 | []                              |          84992 |      5.44  |
# |  86877 | /home/ay/data/0-原始数据集/LAV-DF-Audio/train/088063.wav | train   |       0 |         1 | [[2.8, 3.44]]                   |          84992 |      5.44  |
# |  28357 | /home/ay/data/0-原始数据集/LAV-DF-Audio/train/028885.wav | train   |       1 |         0 | []                              |         166912 |     10.56  |
# | 123192 | /home/ay/data/0-原始数据集/LAV-DF-Audio/train/124789.wav | train   |       1 |         0 | []                              |          87040 |      5.568 |
# | 111776 | /home/ay/data/0-原始数据集/LAV-DF-Audio/train/113209.wav | train   |       0 |         1 | [[2.9, 3.564]]                  |         212992 |     13.44  |
# |   1261 | /home/ay/data/0-原始数据集/LAV-DF-Audio/test/001269.wav  | test    |       0 |         2 | [[6.5, 7.324], [9.324, 10.228]] |         191488 |     12.096 |
# | 105628 | /home/ay/data/0-原始数据集/LAV-DF-Audio/train/106944.wav | train   |       0 |         1 | [[1.4, 2.664]]                  |          87040 |      5.568 |
# |  12182 | /home/ay/data/0-原始数据集/LAV-DF-Audio/dev/012425.wav   | val     |       0 |         1 | [[2.3, 3.234]]                  |         166912 |     10.56  |

class Dataset(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame = None


class LAV_DF_Audio:
    def __init__(self, root_path):
        self.root_path = root_path
        self.data = self.read_metadata()

    def read_metadata(self):
        data = pd.read_json(os.path.join(self.root_path, "metadata.min.json"))
        data["path"] = data["file"].apply(
            lambda x: os.path.join(self.root_path, x.replace("mp4", "wav"))
        )
        data["label"] = data["modify_audio"].apply(lambda x: 0 if x else 1)
        data['fake_periods'] = data.apply(lambda x: x['fake_periods'] if not x['label'] else [], axis=1)
        data['n_fakes'] = data.apply(lambda x: x['n_fakes'] if not x['label'] else 0, axis=1)
        data['split'] = data['split'].replace('dev', 'val')
        data['fps'] = 16_000
        data = data[
            [
                "file",
                "path",
                "split",
                "label",
                "n_fakes",
                "fake_periods",
                "audio_frames",
                "duration",
                "fps"
            ]
        ]

        return data
    
    def get_splits(self, *kargs, **kwargs):
        res = {}
        for split in ['train', 'val', 'test']:
            res[split] = self.data.query(f"split == '{split}'")
        return Dataset(**res)

# + tags=["active-ipynb"]
# lav_df = LAV_DF_Audio(root_path= "/home/ay/data/0-原始数据集/LAV-DF-Audio")
# datas = lav_df.get_splits()

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# %matplotlib inline
# import matplotlib.pyplot as plt
# counts = lav_df.data['audio_frames'].value_counts()
# plt.bar(list(counts.index), list(counts.values))
#
# # 添加标签和标题
# plt.xlabel('Name')
# plt.ylabel('Count')
# plt.title('Frequency of Names')
# plt.show()
