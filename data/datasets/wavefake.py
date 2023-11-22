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

import os
from enum import Enum
from typing import Union
from argparse import Namespace
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

from myutils.common.audio import get_fps_len
from myutils.tools import check_dir, read_file_paths_from_folder, to_list
from myutils.tools.pandas import DF_spliter

# +
import random
from typing import NamedTuple

import numpy as np
# -

from .base import Base


# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from base import Base
# -

class Dataset(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame = None


# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## Dataset Preparation
# -

# 1. Uncompress the wavefake dataset, rename it into 'WaveFake'
# 2. change the folder `WaveFake/common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech` into 'WaveFake/ljspeech_tts'
#     * Inside `WaveFake/ljspeech_tts`, there are directly 16283 audios, but the folder `WaveFake/ljspeech_tts/generated` still have 16283 audios. We delete the `generated` folder.
# 3. Uncompress the LJSeech dataset, rename it into `ljspeech_real` and put it in the `WaveFake` folder.
# 4. Uncompress the JSUT dataset, rename it into `jsut_real` and put it in the `WaveFake` folder.
#
# The folder sturcture of WaveFake is: 
# ```json
# WaveFake
# ├── jsut_multi_band_melgan
# ├── jsut_parallel_wavegan
# ├── jsut_real
# ├── ljspeech_full_band_melgan
# ├── ljspeech_hifiGAN
# ├── ljspeech_melgan
# ├── ljspeech_melgan_large
# ├── ljspeech_multi_band_melgan
# ├── ljspeech_parallel_wavegan
# ├── ljspeech_real
# ├── ljspeech_tts
# ├── ljspeech_waveglow
# └── readme.txt
#
# 12 directories, 1 file
# ```

#
# 每个文件夹下的音频数量如下：
# | trainSet   | method            |   path |
# |:-----------|:------------------|-------:|
# | jsut       | multi_band_melgan |   5000 |
# | jsut       | parallel_wavegan  |   5000 |
# | jsut       | real              |   5000 |
# | ljspeech   | full_band_melgan  |  13100 |
# | ljspeech   | hifiGAN           |  13100 |
# | ljspeech   | melgan            |  13100 |
# | ljspeech   | melgan_large      |  13100 |
# | ljspeech   | multi_band_melgan |  13100 |
# | ljspeech   | parallel_wavegan  |  13100 |
# | ljspeech   | real              |  13100 |
# | ljspeech   | tts               |  16283 |
# | ljspeech   | waveglow          |  13100 |

# ## WaveFake class

# `WaveFake` will read the metadata info for all the audios, and save it (csv format) in the root_path of WaveFake. The examples of the metadata are showed as:
#
# |        | path                                                                                         | trainSet   | method            |   fps |   length |   label | id               |
# |-------:|:---------------------------------------------------------------------------------------------|:-----------|:------------------|------:|---------:|--------:|:-----------------|
# |  15321 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_tts/gen_13607.wav                    | ljspeech   | tts               | 22050 |  3.20435 |       0 | gen_13607        |
# |  91937 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan/LJ024-0083_gen.wav            | ljspeech   | melgan            | 22050 |  3.25079 |       0 | LJ024-0083       |
# |  43707 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_waveglow/LJ018-0097.wav              | ljspeech   | waveglow          | 22050 |  5.7005  |       0 | LJ018-0097       |
# |  75366 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_parallel_wavegan/LJ043-0150_gen.wav  | ljspeech   | parallel_wavegan  | 22050 |  4.69043 |       0 | LJ043-0150       |
# | 121075 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan_large/LJ003-0283_gen.wav      | ljspeech   | melgan_large      | 22050 |  8.85841 |       0 | LJ003-0283       |
# |   4522 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_tts/gen_7735.wav                     | ljspeech   | tts               | 22050 |  6.33905 |       0 | gen_7735         |
# | 106158 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/jsut_multi_band_melgan/BASIC5000_4225_gen.wav | jsut       | multi_band_melgan | 24000 |  8.9875  |       0 | BASIC5000_4225   |
# |   1361 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_tts/gen_14993.wav                    | ljspeech   | tts               | 22050 |  4.51628 |       0 | gen_14993        |
# | 106225 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/jsut_multi_band_melgan/BASIC5000_4962_gen.wav | jsut       | multi_band_melgan | 24000 |  2.85    |       0 | BASIC5000_4962   |
# | 125639 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_multi_band_melgan/LJ033-0121_gen.wav | ljspeech   | multi_band_melgan | 22050 |  4.55111 |       0 | LJ033-0121       |
# |  51593 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_waveglow/LJ031-0180.wav              | ljspeech   | waveglow          | 22050 |  7.53488 |       0 | LJ031-0180       |
# |  90079 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan/LJ021-0099_gen.wav            | ljspeech   | melgan            | 22050 |  8.71909 |       0 | LJ021-0099       |
# |  87734 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_melgan/LJ002-0280_gen.wav            | ljspeech   | melgan            | 22050 |  9.8917  |       0 | LJ002-0280       |
# |  71944 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_hifiGAN/LJ043-0122_generated.wav     | ljspeech   | hifiGAN           | 22050 | 10.0078  |       0 | LJ043-0122erated |
# |  16546 | /usr/local/ay_data/dataset/0-deepfake/WaveFake/ljspeech_real/wavs/LJ031-0084.wav             | ljspeech   | real              | 22050 |  7.87868 |       1 | LJ031-0084       |m

VOCODERs = [
    "melgan",
    "parallel_wavegan",
    "multi_band_melgan",
    "full_band_melgan",
    "hifiGAN",
    "melgan_large",
    "waveglow",
]
TRAINSETs = ["ljspeech", "jsut"]


# + editable=true slideshow={"slide_type": ""}
class WaveFake(Base):
    def __init__(self, root_path="/usr/local/ay_data/dataset/0-deepfake/WaveFake"):
        """
        When crate a entry of WaveFake, it will read all the metadatas from the root_path

        Args:
            root_path: the path of WaveFake dataset. Note that the path must contain "/WaveFake/"
        """

        self.root_path = root_path if not root_path.endswith("/") else root_path[:-2]
        self.data = self.read_metadata(self.root_path)
        self.data["vocoder_label"] = self.data["method"].apply(
            lambda x: 0 if (x == "real" or x not in VOCODERs) else VOCODERs.index(x) + 1
        )

        self.train_sets = ["ljspeech", "jsut"]

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
        data["path"] = wav_paths

        ## Step 2. obtain the TTS method and their corresponding train set
        ## train set contain:  ["ljspeech", "jsut"]
        ## TTS method contain: ['multi_band_melgan', 'parallel_wavegan', 'full_band_melgan',
        ##                     'hifiGAN', 'melgan', 'melgan_large', 'waveglow', 'real', 'tts']
        ##         where 'real' means that the wav is real and is from the origianl train set for training TTS.

        def get_trainSet_method(path):
            trainSet_method = path.split("/WaveFake/")[1].split("/")[0]
            trainSet = trainSet_method.split("_")[0]
            method = trainSet_method.split(trainSet + "_")[1]
            return [trainSet, method]

        data[["trainSet", "method"]] = data.apply(
            lambda x: tuple(get_trainSet_method(x["path"])),
            axis=1,
            result_type="expand",
        )

        pandarallel.initialize(progress_bar=True, nb_workers=20)
        data[["fps", "length"]] = data.parallel_apply(
            lambda x: tuple(get_fps_len(x["path"])), axis=1, result_type="expand"
        )

        # Output the nubmer of audios for each sub-folder.
        # print(data.groupby(['trainSet', 'method']).count().reset_index().to_markdown(index=False))

        ## Step 3. save the metadatas
        data["label"] = data["path"].apply(
            lambda x: 1 if "ljspeech_real" in x or "jsut_real" in x else 0
        )
        data["id"] = data["path"].apply(
            lambda x: os.path.basename(x)
            .replace(".wav", "")
            .replace("_generated", "")
            .replace("_gen", "")
        )

        data.to_csv(data_path, index=False)
        return data

    def _get_sub_data(self, trainset, method):
        """
        Given the trainset of Vocoders and the vocoder method, return the subdata
        Args:
            trainSet: the dataset for training the Vocoders
            method: the vocoder method
        """
        # print(trainSet, method)
        if isinstance(trainset, int):
            trainset = TRAINSETs[trainset]
        if isinstance(method, int):
            method = VOCODERs[method]

        data = self.data
        sub_data = data[(data["trainSet"] == trainset) & (data["method"] == method)]
        return sub_data.reset_index(drop=True)

    def get_sub_data(self, trainset: [list, str], methods: [list, str], contain_real=True) -> pd.DataFrame:
        trainset = to_list(trainset)
        methods = to_list(methods)
        if contain_real:
            methods = methods + ['real']
        data = []
        for _trainset in trainset:
            for _method in methods:
                _data = self._get_sub_data(_trainset, _method)
                data.append(_data)
        data = pd.concat(data).reset_index(drop=True)
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

# + tags=["active-ipynb", "style-student"]
# dataset = WaveFake(root_path="/usr/local/ay_data/dataset/0-deepfake/WaveFake")
# data = dataset.get_sub_data(trainset=0, methods=[0, 1, 2, 3, 4, 5, 6])
# splits = [64_000, 16_000, 24_800]
# datas = dataset.split_data(data, splits)
