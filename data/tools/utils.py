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

# +
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from einops import rearrange
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


# +
def padding_audio(
    tensor: Tensor,
    target: int,
    padding_method: str = "zero",
    padding_position: str = "tail",
) -> Tensor:
    c, t = tensor.shape
    padding_size = target - t
    pad = _get_padding_pair(padding_size, padding_position)

    
    if padding_method == "zero":
        return F.pad(tensor, pad=pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c -> 1 c t")
        tensor = F.pad(tensor, pad=pad, mode="replicate")
        return rearrange(tensor, "1 c t -> t c")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def _get_padding_pair(padding_size: int, padding_position: str) -> List[int]:
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError(
            "Wrong padding position. It should be zero or tail or average."
        )
    return pad


# +
def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores

