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

# + editable=true slideshow={"slide_type": ""}
"""Common preprocessing functions for audio data."""
import functools
import logging
import math
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.functional import apply_codec
import pandarallel

# +
from typing import Any, Callable, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
# -

from .dataset import BaseDataset
from .utils import ioa_with_anchors, iou_with_anchors, padding_audio


class Localization_DS(BaseDataset):
    def __init__(
        self,
        data,
        sample_rate: int = 16_000,
        normalize: bool = True,
        trim: bool = False,
        # custome args
        max_wave_length: int = 16_000 * 20,
        transform=torch.nn.Identity(),
        is_training=False,
    ):
        super().__init__(
            data=data, sample_rate=sample_rate, normalize=normalize, trim=trim
        )

        self.data = data
        self.max_wave_length = max_wave_length
        self.transform = transform
        self.is_training = is_training
        self.max_duration = 64
        
        
        
    def read_metadata(self, index: int) -> dict:
        item = self.data.iloc[index]
        keys = item.keys()
        res = {"sample_rate": self.sample_rate}

        label_path = item["path"].replace(".wav", ".npz")
        if os.path.exists(label_path):
            label = np.load(label_path)
            bm_label = torch.from_numpy(label['bm_label'])
            frame_label = torch.from_numpy(label['frame_label'])
        else:
            bm_label, frame_label = self.gen_label(index)

        res['bm_label'] = bm_label
        res['frame_label'] = frame_label
        res['frames'] = int(item["audio_frames"] / 16000 * 40)
        res["name"] = item["file"]
        res['fake_periods'] = item['fake_periods']
        return res

    def __getitem__(self, index: int):
        waveform = self.read_audio(index)
        waveform = padding_audio(waveform, target=self.max_wave_length)
        waveform = self.transform(waveform)

        waveform = self._get_log_mel_spectrogram(waveform)
        
        res = self.read_metadata(index)
        res["audio"] = waveform
        return res

    def _get_log_mel_spectrogram(self, audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=201, n_mels=64)
        spec = torch.log(ms(audio[0, :]) + 0.01)
        assert spec.shape == (64, 3200), "Wrong log mel-spectrogram setup in Dataset"
        return spec

    
    def _get_audio_label(
        self, audio_length, fake_periods
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        corrected_second = audio_length / self.sample_rate  # number of audio seconds
        audio_frames = int(
            audio_length / 16000 * 40
        )  # number of audio clips (25ms per clip, thus 40 frames/second)
        temporal_gap = 1 / audio_frames

        #############################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        for j in range(len(fake_periods)):
            tmp_start = max(min(1, fake_periods[j][0] / corrected_second), 0)
            tmp_end = max(min(1, fake_periods[j][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ###########################################################################
        # generate R_s and R_e
        gt_bbox = torch.tensor(gt_bbox)
        if len(gt_bbox) > 0:
            gt_xmins = gt_bbox[:, 0]
            gt_xmaxs = gt_bbox[:, 1]
        else:
            gt_xmins = np.array([])
            gt_xmaxs = np.array([])

        ###########################################################################

        gt_iou_map = torch.zeros([self.max_duration, audio_frames])

        if len(gt_bbox) > 0:
            for begin in range(audio_frames):
                for duration in range(self.max_duration):
                    end = begin + duration
                    if end > audio_frames:
                        break
                    gt_iou_map[duration, begin] = torch.max(
                        iou_with_anchors(
                            begin * temporal_gap,
                            (end + 1) * temporal_gap,
                            gt_xmins,
                            gt_xmaxs,
                        )
                    )
                    # [i, j]: Start in i, end in j.

        ############################################################################
        max_wave_frames = int(self.max_wave_length / 16000 * 40)
        gt_iou_map = F.pad(
            gt_iou_map.float(),
            pad=[0, max_wave_frames - audio_frames, 0, 0],
        )
        
        
        bm_label = gt_iou_map
        frame_label = torch.ones(max_wave_frames)
        for begin, end in fake_periods:
            begin = int(begin * 40)
            end = int(end * 40)
            frame_label[begin: end] = 0
        
        return bm_label, frame_label
    
    def gen_label(self, index:int, overwrite=False)->Tuple[Tensor, Tensor]:
        item = self.data.iloc[index]
        label_path = item["path"].replace(".wav", ".npz")
        if not overwrite and os.path.exists(label_path):
            label = np.load(label_path)
            bm_label = torch.from_numpy(label['bm_label'])
            frame_label = torch.from_numpy(label['frame_label'])
            return bm_label, frame_label
        
        bm_label, frame_label = self._get_audio_label(
                item["audio_frames"], fake_periods=item["fake_periods"]
            )
        np.savez_compressed(label_path, bm_label=bm_label, frame_label=frame_label)
        return bm_label, frame_label
    
    
    def gen_labels(self, overwrite=False):
        from pandarallel import pandarallel

        pandarallel.initialize(progress_bar=True, nb_workers=15)
        data = pd.DataFrame()
        data['id'] = list(range(len(self.data)))
        data['id'].parallel_apply(lambda x: self.gen_label(x, overwrite=overwrite))
        # for _id in tqdm(range(len(self.data))):
            # self.gen_label(_id)

# `gt_iou_map[duration, begin]` 表示 以 `begin`作为帧数的开始，以`begin + duration`作为帧数的结束，这个范围里帧和真实fake区间的IoU分数。

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# import sys
#
# sys.path.append("/home/ay/zky/Coding/0-Audio/data/datasets")
#
# from LAV_DF import LAV_DF_Audio
#
# lav_df = LAV_DF_Audio(root_path="/home/ay/data/0-原始数据集/LAV-DF-Audio")
# datas = lav_df.get_splits()
#
# ds = LAVDF(datas.val)
