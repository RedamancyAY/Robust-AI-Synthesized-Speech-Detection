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
from argparse import Namespace
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

from .base import Base

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from base import Base
# -

# ---

# ## 数据集

# There are 63098 and 55283 utterances in the Chinese and English subsets, respectively. 
#
#
# <table>
#     <tr>
#   		 <td></td> 
#       	 <td colspan="3">English</td>   
#       	 <td colspan="3">Chinese</td>    
#     </tr>
#     <tr>
#   		 <td></td> 
#   		 <td>Train Set</td> 
#   		 <td>Dev Set</td> 
#   		 <td>Eval Set</td> 
#   		 <td>Train Set</td> 
#   		 <td>Dev Set</td> 
#   		 <td>Eval Set</td> 
#     </tr>
#     <tr>
#   		 <td>Bona-fide</td> 
#   		 <td>5129</td> 
#   		 <td>3049</td> 
#   		 <td>4306</td> 
#   		 <td>9000</td> 
#   		 <td>6109</td> 
#   		 <td>6109</td> 
#     </tr>
#     <tr>
#   		 <td>Spoofed</td> 
#   		 <td>17412</td> 
#   		 <td>10503</td> 
#   		 <td>14884</td> 
#   		 <td>17850</td> 
#   		 <td>12015</td> 
#   		 <td>12015</td> 
#     </tr>
#     <tr>
#   		 <td>Total</td> 
#   		 <td>22541</td>
#   		 <td>13552</td> 
#   		 <td>19190</td> 
#   		 <td>26850</td> 
#   		 <td>18124</td> 
#   		 <td>18124</td> 
#     </tr>
# </table>

VOCODERS = [
    "FastSpeech2",
    "StarGAN",
    "Tacotron",
    "baidu",
    "baidu_en",
    "fs2",
    "fs2mtts",
    "hifigan",
    "mbmelgan",
    "nvcnet",
    "nvcnet-cn",
    "pwg",
    "starganv2",
    "tacotron",
    "vits",
    "xunfei",
]


class DECRO(Base):
    def __init__(self, root_path="/home/ay/data/DATA/dataset/0-audio/DECRO"):
        """
        DECRO is from the paper 'Transferring Audio Deepfake Detection Capability across Languages'

        Args:
            root_path: the path of dataset. Note that the path must contain "/DECRO/"
        """

        self.root_path = root_path if not root_path.endswith("/") else root_path[:-2]
        self.data = self.read_metadata(self.root_path)
        self.data["path"] = self.data["filename"].apply(
            lambda x: os.path.join(root_path, x)
        )
        self.data['vocoder_label'] = self.data['method'].apply(
            lambda x: 0 if not x in VOCODERS else VOCODERS.index(x) + 1
        )

    def read_metadata_from_txt(self, root_path):
        datas = []
        for txt_file in [
            "ch_dev",
            "ch_eval",
            "ch_train",
            "en_dev",
            "en_eval",
            "en_train",
        ]:
            with open(os.path.join(root_path, txt_file + ".txt"), "r") as f:
                lines = f.readlines()

                for line in lines:
                    line_splits = line.strip().split(" ")
                    filename = f"{txt_file}/{line_splits[1]}.wav"
                    method = line_splits[3]
                    label = 1 if line_splits[-1] == "bonafide" else 0
                    language = txt_file.split("_")[0]
                    split = (
                        txt_file.split("_")[1]
                        .replace("dev", "val")
                        .replace("eval", "test")
                    )
                    datas.append([filename, method, label, language, split])

        data = pd.DataFrame(
            datas, columns=["filename", "method", "label", "language", "split"]
        )
        return data

    def read_metadata(self, root_path):
        """
        read metadatas of audio files from the root_path. We frist extract the metadata
        from the given txt files, then get the fps and length for each audio.

        Args:
            root_path: the root_path for the DECRO.

        """

        data_path = os.path.join(root_path, "dataset_info.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)

        ## Step 1. read all audio paths
        data = self.read_metadata_from_txt(root_path)
        data["path"] = data["filename"].apply(lambda x: os.path.join(root_path, x))
        data = self.read_fps_length(data)
        data.to_csv(data_path, index=False)
        return data

    def get_splits(self, language="en"):
        """
            Get train/val/test splits according to the language.

        Args:
            language: 'en' or 'ch'

        Returns:
            Namespace(train, val, test)
        """

        assert language in ["en", "ch"]
        data = self.data.query(f'language == "{language}"')

        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        return Namespace(
            train=sub_datas[0],
            test=sub_datas[-1],
            val=None if len(sub_datas) == 2 else sub_datas[1],
        )

# + tags=["active-ipynb", "style-student"]
# model = DECRO()
# data = model.get_splits(language="ch")
#
# data.train
