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
"""Common preprocessing functions for audio data."""
import functools
import logging
import math
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.functional import apply_codec

LOGGER = logging.getLogger(__name__)


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


# -

# ## Split data by audio length

def audio_data_split(data, len_clip, len_sep):
    data["sec"] = data.apply(lambda x: x["length"] // x["fps"], axis=1)
    data1 = data.query(f"length <= {len_clip}").reset_index(drop=True)
    data1["start"] = 0
    data1["end"] = data1["length"]

    data2 = data.query(f"length > {len_clip}").reset_index(drop=True)

    max_sec = data2["sec"].max()
    min_sec = data2["sec"].min()

    _datas = []
    for s in range(min_sec, max_sec + 1):
        _data = data2.query(f"sec == {s}")
        _d = []
        for i in range(s):
            _t = _data.copy()
            _t["start"] = i * len_sep
            _t["end"] = i * len_sep + len_clip
            _d.append(_t)
        _data = pd.concat(_d).reset_index(drop=True).query("start < length")
        _data["end"] = _data.apply(lambda x: min(x["end"], x["length"]), axis=1)
        _datas.append(_data)

    data2 = pd.concat(_datas).reset_index(drop=True)
    return pd.concat([data1, data2]).reset_index(drop=True)


# ## over sample

def over_sample_dataset(data):
    n_fake = len(data[data["label"] == 0])
    n_real = len(data[data["label"] == 1])
    if n_fake == n_real:
        return data
    if n_fake > n_real:
        sampled = data[data["label"] == 1].sample(n=n_fake - n_real, replace=True)
        balanced_data = pd.concat([data, sampled])
    else:
        sampled = data[data["label"] == 0].sample(n=n_real - n_fake, replace=True)
        balanced_data = pd.concat([data, sampled])

    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    return balanced_data


# # Dataset

class BaseDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory.

    Args:
        data: A pandas dataframe, whose 'path' column is the path for each audio file.
        sample_rate: The used sample rate for the audio
        amount: default None. If not none, it means the number of used audio
        normalize: default True.
        trim: default True. trim all silence that is longer than 0.2s and louder than 1% volume
        phone_call: default False.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        data,
        sample_rate: int = 16_000,
        normalize: bool = True,
        trim: bool = False,
    ) -> None:
        super().__init__()

        self.data = data
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.trim = trim

    def read_metadata(self, index: int) -> dict:
        item = self.data.iloc[index]
        keys = item.keys()
        res = {"sample_rate": self.sample_rate}
        if "label" in keys:
            res["label"] = item["label"]
        if "name" in keys:
            res["name"] = item["name"]
        else:
            res["name"] = item["path"]
        if "vocoder_label" in keys:
            res["vocoder_label"] = item["vocoder_label"]
        if "emotion_label" in keys:
            res["emotion_label"] = item["emotion_label"]
        return res

    def read_audio(self, index: int) -> Tuple[torch.Tensor, int, int]:
        item = self.data.iloc[index]

        path = item["path"]
        fps = item["fps"]

        # read audio ath self.sampling_rate
        if fps != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )
        else:
            waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        # trim the salience of audio
        if self.trim:
            (
                waveform_trimmed,
                sample_rate_trimmed,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE
            )

            if waveform_trimmed.size()[1] > 0:
                waveform = waveform_trimmed
                sample_rate = sample_rate_trimmed
        return waveform

    def __getitem__(self, index: int) -> dict:
        waveform = self.read_audio(index)
        res = self.read_metadata(index)
        res["audio"] = waveform
        return res

    def __len__(self) -> int:
        return len(self.data)


class WaveDataset(BaseDataset):
    def __init__(
        self,
        data,
        sample_rate: int = 16_000,
        normalize: bool = True,
        trim: bool = False,
        # custome args
        max_wave_length: int = 64600,
        transform=None,
        is_training=False,
        **kwargs,
    ) -> None:
        super().__init__(
            data=data, sample_rate=sample_rate, normalize=normalize, trim=trim
        )
        self.is_training = is_training
        self.transform = transform
        self.max_wave_length = max_wave_length

    def check_length(self, waveform):
        waveform_len = waveform.shape[-1]

        if self.max_wave_length == -1:
            return waveform

        # don't need to pad
        if waveform_len >= self.max_wave_length:
            if self.is_training:
                start = random.randint(0, waveform_len - self.max_wave_length)
            else:
                start = (waveform_len - self.max_wave_length) // 2
            return waveform[:, start : start + self.max_wave_length]

        # need to pad
        num_repeats = int(math.ceil(self.max_wave_length / waveform_len))
        padded_waveform = torch.tile(waveform, (1, num_repeats))[
            :, : self.max_wave_length
        ]
        return padded_waveform

    
    def __getitem__(self, index: int):
        waveform = self.read_audio(index)
        waveform = self.check_length(waveform)
        if self.transform is not None:
            waveform = self.transform(waveform)

        res = self.read_metadata(index)
        res["audio"] = waveform
        return res


class FeatureDataset(WaveDataset):
    def __init__(
        self,
        data,
        sample_rate: int = 16_000,
        normalize: bool = True,
        trim: bool = False,
        # custome args
        feature=None,
        max_wave_length: int = 48000,
        max_feature_frames=300,
        transform=None,
        is_training=False,
    ) -> None:
        super().__init__(
            data=data,
            sample_ratel=sample_rate,
            normalize=normalize,
            triml=trim,
            max_wave_length=max_wave_length,
            transform=transform,
            is_training=is_training,
        )
        self.feature = feature
        self.max_feature_frames = max_feature_frames

    def check_frames(self, wave_feature):
        L = wave_feature.shape[0]
        if L < self.max_feature_frames:
            mul = int(math.ceil(self.max_feature_frames / L))
            wave_feature = torch.tile(wave_feature, (mul, 1))[: self.max_feature_frames]
        else:
            if self.is_training:
                start_p = random.randint(0, L - self.max_feature_frames)
            else:
                start_p = (L - self.max_feature_frames) // 2
            wave_feature = wave_feature[start_p : start_p + self.max_feature_frames]

        return wave_feature

    def __getitem__(self, index: int):
        path = self.data.iloc[index]["path"]

        # feature_name = self.feature.__class__.__name__
        # feature_path = path.replace(".wav", f"-{feature_name}.npy")
        # if os.path.exists(feature_path):
        #     wave_feature = torch.from_numpy(np.load(feature_path))
        # else:
    
        waveform = self.read_audio(index)
        waveform = self.check_length(waveform)
        if self.transform is not None:
            waveform = self.transform(waveform)
        wave_feature = self.feature(waveform)
        # np.save(feature_path, wave_feature)
        wave_feature = self.check_frames(wave_feature)

        res = self.read_metadata(index)
        res["audio"] = wave_feature
        return res


class AudioDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory.

    Args:
        data: A pandas dataframe, whose 'path' column is the path for each audio file.
        sample_rate: The used sample rate for the audio
        amount: default None. If not none, it means the number of used audio
        normalize: default True.
        trim: default True. trim all silence that is longer than 0.2s and louder than 1% volume
        phone_call: default False.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        data,
        sample_rate: int = 16_000,
        normalize: bool = True,
        trim: bool = False,
        phone_call: bool = False,
        audio_feature=None,
        max_feature_frames=None,
        # post processing
        len_clip: int = 64600,
        len_sep: int = 48000,
        audio_split=False,
        over_sample=False,
        random_cut=False,
        transform=None,
        test=False,
    ) -> None:
        super().__init__()

        if audio_split:
            self.data = audio_data_split(data, len_clip, len_sep)
        else:
            self.data = data
        if over_sample:
            self.data = over_sample_dataset(self.data)

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.phone_call = phone_call

        # post processing
        self.len_clip = len_clip
        self.len_sep = len_sep
        self.random_cut = random_cut
        self.transform = transform

    def read_audio(self, index: int) -> Tuple[torch.Tensor, int, int]:
        item = self.data.iloc[index]

        path = item["path"]
        fps = item["fps"]

        if fps != self.sample_rate:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
                path, [["rate", f"{self.sample_rate}"]], normalize=self.normalize
            )
        else:
            waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if self.trim:
            (
                waveform_trimmed,
                sample_rate_trimmed,
            ) = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, SOX_SILENCE
            )

            if waveform_trimmed.size()[1] > 0:
                waveform = waveform_trimmed
                sample_rate = sample_rate_trimmed

        if self.phone_call:
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform,
                sample_rate,
                effects=[
                    ["lowpass", "4000"],
                    [
                        "compand",
                        "0.02,0.05",
                        "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                        "-8",
                        "-7",
                        "0.05",
                    ],
                    ["rate", "8000"],
                ],
            )
            waveform = apply_codec(waveform, sample_rate, format="gsm")

        if "start" in item.keys():
            s = int(item["start"])
            e = int(item["end"])
            waveform = waveform[:, s:e]

        res = {"audio": waveform, "sample_rate": sample_rate, "name": item["path"]}
        if "label" in item.keys():
            res["label"] = item["label"]
        return res

    def cut_audio(self, waveform):
        waveform_len = waveform.shape[-1]

        # don't need to pad
        if waveform_len >= self.len_clip:
            if self.random_cut:
                start = random.randint(0, waveform_len - self.len_clip)
            else:
                start = 0
            return waveform[:, start : start + self.len_clip]

        # need to pad
        num_repeats = int(self.len_clip / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, : self.len_clip]
        return padded_waveform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        res = self.read_audio(index)  # 1. read audio

        # 2. post precessing
        if self.len_clip > 0:
            res["audio"] = self.cut_audio(res["audio"])

        if self.transform is not None:
            res["audio"] = self.transform(res["audio"])

        return res

    def __len__(self) -> int:
        return len(self.data)

# + tags=["active-ipynb", "style-solution"]
# from datasets import WaveFake
#
# wave = WaveFake("/usr/local/ay_data/dataset/0-deepfake/WaveFake")
# ds = AudioDataset(wave.data, cut=500000)
# ds[0][0].shape
