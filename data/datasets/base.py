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

# + editable=true slideshow={"slide_type": ""}
import json
import os
from enum import Enum
from typing import Union

import pandas as pd
from myutils.common.audio import get_fps_len
from myutils.tools import check_dir, read_file_paths_from_folder, to_list
from myutils.tools.pandas import DF_spliter
from pandarallel import pandarallel
from tqdm.auto import tqdm


# -

# ---

# + editable=true slideshow={"slide_type": ""}
class Base:
    def __init__(self, root_path):
        """
        When crate a entry of WaveFake, it will read all the metadatas from the root_path

        Args:
            root_path: the path of WaveFake dataset. Note that the path must contain "/WaveFake/"
        """

        self.root_path = root_path if not root_path.endswith("/") else root_path[:-2]
        self.data = self.read_metadata(self.root_path)

    def read_emotion_label(self):
        emotion_path = os.path.join(self.root_path, "emotion.json")

        # read json file and convert to Dataframe
        with open(emotion_path, "r") as file:
            json_data = json.load(file)
            res = dict(json_data)

        path = res.keys()
        emotion = [res[p] for p in path]
        data = pd.DataFrame([path, emotion]).transpose()
        data.columns = ["path", "emotion_label"]

        dataset_name = self.root_path.split("/")[-1]
        data["path"] = data["path"].apply(
            lambda x: os.path.join(self.root_path, x.split(dataset_name + "/")[1])
        )
        self.data = pd.merge(self.data, data, on="path")

    def read_fps_length(self, data: pd.DataFrame) -> pd.DataFrame:
        pandarallel.initialize(progress_bar=True, nb_workers=20)
        data[["fps", "length"]] = data.parallel_apply(
            lambda x: tuple(get_fps_len(x["path"])), axis=1, result_type="expand"
        )
        return data

    def read_metadata(self, root_path):
        """
        read all the metadatas of audio files from the root_path
        """
        raise NotImplementedError

    def split_data(self, data: pd.DataFrame = None, splits=[0.6, 0.2, 0.2]):
        if data is None:
            data = self.data
        sub_datas = DF_spliter.split_df(data, splits)

        return Dataset(
            train=sub_datas[0],
            test=sub_datas[-1],
            val=None if len(splits) == 2 else sub_datas[1],
        )
