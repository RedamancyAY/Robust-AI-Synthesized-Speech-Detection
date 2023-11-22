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

import os
import random
from enum import Enum
from typing import NamedTuple, Union
from argparse import Namespace
import numpy as np
import pandas as pd
from myutils.common.audio import get_fps_len
from myutils.tools import check_dir, read_file_paths_from_folder, to_list
from myutils.tools.pandas import DF_spliter
from pandarallel import pandarallel
from tqdm.auto import tqdm

from .base import Base


# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from base import Base
# -

class Dataset(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame = None


# ---

# ## 数据集

# | method            |   Number |
# |:------------------|-----------:|
# | diffwave          |      13201 |
# | gt                |      13201 |
# | melgan            |      13201 |
# | parallel_wave_gan |      13201 |
# | wavegrad          |      13201 |
# | wavenet           |      13201 |
# | wavernn           |      13201 |

# |       | filename                                            | method            |   fps |       length |   label |                        id | path   |
# |------:|:----------------------------------------------------|:------------------|------:|-------------:|--------:|--------------------------:|:-------|
# | 38796 | melgan/8580_287363_000011_000000_gen.wav            | melgan            | 24000 | 266240       |       0 | 8580_287363_000011_000000 | ...    |
# | 24380 | gt/8014_112602_000001_000001.wav                    | gt                | 24000 | 308640       |       1 | 8014_112602_000001_000001 | ...    |
# | 92137 | wavernn/87_121553_000211_000000_gen.wav             | wavernn           | 24000 | 158400       |       0 |   87_121553_000211_000000 | ...    |
# | 58298 | wavegrad/4195_186237_000007_000001_gen.wav          | wavegrad          | 24000 |      7.4625  |       0 | 4195_186237_000007_000001 | ...    |
# | 11779 | diffwave/8123_275216_000052_000002_gen.wav          | diffwave          | 24000 |     11.4773  |       0 | 8123_275216_000052_000002 | ...    |
# | 65071 | wavegrad/8465_246947_000013_000000_gen.wav          | wavegrad          | 24000 |      5.2625  |       0 | 8465_246947_000013_000000 | ...    |
# |  8286 | diffwave/6272_70191_000024_000006_gen.wav           | diffwave          | 24000 |      8.87467 |       0 |  6272_70191_000024_000006 | ...    |
# | 52573 | parallel_wave_gan/8838_298545_000023_000001_gen.wav | parallel_wave_gan | 24000 | 171008       |       0 | 8838_298545_000023_000001 | ...    |
# |  8561 | diffwave/6415_111615_000019_000003_gen.wav          | diffwave          | 24000 |     10.4747  |       0 | 6415_111615_000019_000003 | ...    |
# | 38585 | melgan/8324_286681_000012_000002_gen.wav            | melgan            | 24000 | 229888       |       0 | 8324_286681_000012_000002 | ...    |
#

# <center><img src="https://raw.githubusercontent.com/RedamancyAY/CloudImage/main/imgCleanShot 2023-06-30 at 16.51.10@2x.png" width="800" alt="CleanShot 2023-06-30 at 16.51.10@2x"/></center>

# 共92407个音频样本，共有6个vocoders。

VOCODERs = [
    "parallel_wave_gan",
    "diffwave",
    "wavenet",
    "wavernn",
    "melgan",
    "wavegrad",
]


class LibriSeVoc(Base):
    def __init__(self, root_path):
        """
        LibriSeVoc is from the paper 'AI-Synthesized Voice Detection Using Neural Vocoder Artifacts'

        Args:
            root_path: the path of dataset. Note that the path must contain "/LibriSeVoc/"
        """

        self.root_path = root_path if not root_path.endswith("/") else root_path[:-2]
        self.data = self.read_metadata(self.root_path)
        self.data["path"] = self.data["filename"].apply(
            lambda x: os.path.join(root_path, x)
        )
        self.data["id"] = self.data["id"].apply(lambda x: x.replace("_gen", ""))
        self.data["vocoder_label"] = self.data["method"].apply(
            lambda x: 0 if x == "gt" else VOCODERs.index(x) + 1
        )


        # self.read_emotion_label()

    
    def read_metadata(self, root_path):
        """
        read all the metadatas of audio files from the root_path
        """

        data_path = os.path.join(root_path, "dataset_info.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)

        ## Step 1. read all audio paths
        wav_paths = read_file_paths_from_folder(root_path, exts=["wav"])
        data = pd.DataFrame()
        data["filename"] = [path.split("/LibriSeVoc/")[1] for path in wav_paths]
        print(data["filename"])

        data["method"] = data["filename"].apply(lambda x: os.path.split(x)[0])
        pandarallel.initialize(progress_bar=True, nb_workers=20)
        data[["fps", "length"]] = data.parallel_apply(
            lambda x: tuple(get_fps_len(os.path.join(root_path, x["filename"]))),
            axis=1,
            result_type="expand",
        )

        # Output the nubmer of audios for each sub-folder.
        # print(data.groupby(['trainSet', 'method']).count().reset_index().to_markdown(index=False))

        ## Step 3. save the metadatas
        data["label"] = data["method"].apply(lambda x: 1 if x == "gt" else 0)
        data["id"] = data["filename"].apply(
            lambda x: os.path.basename(x).replace(".wav", "").replace("_gen", "")
        )

        data.to_csv(data_path, index=False)
        return data

    def get_sub_data(self, methods: [list, str], contain_real=True) -> pd.DataFrame:
        methods = to_list(methods)
        methods = [VOCODERs[x] for x in methods]
        if contain_real:
            methods = methods + ['gt']
            
        data = self.data[self.data["method"].isin(methods)].reset_index(drop=True)
        return data

    def split_data(
        self,
        data: pd.DataFrame = None,
        splits=[0.6, 0.2, 0.2],
        refer="id",
        return_list=False,
    ):
        if data is None:
            data = self.data

        if refer is None:
            sub_datas = DF_spliter.split_df(data, splits)
        else:
            sub_datas = DF_spliter.split_by_number_and_column(data, splits, refer=refer)

        if return_list:
            return sub_datas

        return Namespace(
            train=sub_datas[0],
            test=sub_datas[-1],
            val=None if len(splits) == 2 else sub_datas[1],
        )

# + tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = LibriSeVoc("/home/ay/data/DATA/dataset/0-audio/LibriSeVoc")
# datas = model.split_data(model.data)
