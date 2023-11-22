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

# +
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils.torch.nn import Conv2p1D, LayerNorm, PositionEmbedding
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn


# -

def weight_init(m):

    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.Conv3d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# ## AdaptiveConv1d

class AdaptiveConv1d(nn.Module):
    """
        Given the channel numbers, kernel size, stride, the reduction percentage of the feature length,
        this module can adaptively calcuate the padding size,
            and in the transpose conv, calcuate the output_padding size,

    Args:
        n_dim: channel number
        kernel_size: kernel size for conv
        stride: stride for conv
        reduction: the reduction percentage (>1) of the feature length, `1` denotes not change.
        conv_transpose: whether add the weights and bias for transpose conv
    """

    def __init__(
        self,
        n_dim,
        kernel_size,
        stride,
        reduction,
        reverse_conv='upsample',
        groups=1,
        **kwargs
    ):
        super().__init__()
        self.register_parameter(
            param=nn.Parameter(torch.randn(n_dim, n_dim // groups, kernel_size)),
            name="weights1",
        )
        self.register_parameter(param=nn.Parameter(torch.randn(n_dim)), name="bias1")

        self.post_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1)
        )
        
        self.reverse_type = reverse_conv
        if reverse_conv is not None:
            if reverse_conv == 'convT':
                self.register_parameter(
                    param=nn.Parameter(torch.randn(n_dim, n_dim // groups, kernel_size)),
                    name="weights2",
                )
                self.register_parameter(
                    param=nn.Parameter(torch.randn(n_dim)), name="bias2"
                )
            else:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=reduction),
                    nn.Conv1d(n_dim, n_dim, kernel_size=5, stride=1, padding=2)
                )

        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction = reduction
        self.groups = groups

    def conv(self, x, reduction=None, **kwargs):
        length = x.shape[-1]
        out_length = length // self.reduction

        p = 0
        _out_length = math.floor((length - self.kernel_size + 2 * p) / self.stride + 1)
        while _out_length != out_length:
            p += 1
            _out_length = math.floor(
                (length - self.kernel_size + 2 * p) / self.stride + 1
            )
            # print(_out_length, out_length, p)
        self.p = p
        y = F.conv1d(
            x,
            self.weights1,
            bias=self.bias1,
            stride=self.stride,
            padding=p,
            groups=self.groups,
        )
        y = self.post_conv(y)
        return y

    def conv_transpose(self, y):
        length = y.shape[-1]
        out_length = length * self.reduction

        out_padding = 0
        _out_length = (
            (length - 1) * self.stride - 2 * self.p + self.kernel_size + out_padding
        )
        while _out_length != out_length:
            out_padding += 1
            _out_length = (
                (length - 1) * self.stride - 2 * self.p + self.kernel_size + out_padding
            )
        x = F.conv_transpose1d(
            y,
            self.weights2,
            bias=self.bias2,
            stride=self.stride,
            padding=self.p,
            output_padding=out_padding,
            groups=self.groups,
        )
        return x

    def forward(self, x):
        return self.conv(x)

    def reverse(self, x):
        if self.reverse_type == 'convT':
            return self.conv_transpose(x)
        else:
            return self.upsample(x)


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-activity"]
# x = torch.randn(2, 128, 64000)
#
# model = AdaptiveConv1d(n_dim=128, kernel_size=25, stride=25, reduction=25, groups=1, conv_transpose='convT')
# y = model(x)
# print(y.shape)
#
# z = model.reverse(y)
# print(z.shape)
# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-activity"]
# model = AdaptiveConv1d(n_dim=128, kernel_size=25, stride=25, reduction=25, groups=1, conv_transpose='upsample')
# y = model(x)
# print(y.shape)
# z = model.upsample(y)
# print(z.shape)
# -
# ## Self-attention

# +
def make_relative_position(q_seq_len, seq_len, k):
    Q = torch.arange(q_seq_len)[:, None]  # q_seq_len, 1
    # it might be key or value
    S = torch.arange(seq_len)[None, :]  # 1, seq_len
    # max(-k,min(j-i,k)) - j is seq_len of key/value and i is seq_len of query
    rp = torch.clip(S - Q, -k, k)  # q_seq_len, seq_len
    # + k
    out = rp + k
    return out


# batch, h와는 무관
class RelativePositionEmbedding(nn.Module):
    def __init__(self, max_k, embed_dim, n_head):
        super().__init__()
        self.max_k = max_k
        self.d_k = embed_dim // n_head
        self.emb = nn.Embedding(2 * max_k + 1, self.d_k)

        self.cache_pos = {}
    
    def _make_relative_postion(self, seq_len):
        if not seq_len in self.cache_pos:
            self.cache_pos[seq_len] = make_relative_position(seq_len, seq_len, self.max_k)
        return self.cache_pos[seq_len].to(self.emb.weight.device)
    
    def forward(self, seq_len):
        # relative_position
        """
        relative position
        shape : seq_len(query), seq_len(key or value)
        """
        out = self._make_relative_postion(seq_len)
        out = self.emb.forward(out)
        return out


# -

class Multi_Head_Attention(nn.Module):
    def __init__(
        self,
        max_k,
        embed_dim,
        num_heads=1,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.PE = RelativePositionEmbedding(max_k, embed_dim, num_heads)
        self.num_heads = num_heads
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False), nn.Dropout(dropout)
        )
        self.apply(self._init_weights)
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _multi_head_attention(self, q, k, v):
        q, k, v = map(
            lambda mat: rearrange(mat, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )
        scale = q.shape[-1] ** -0.5
        qkT = einsum("b n d, b m d->b n m", q, k) * scale

        # relative positive embedding
        rpe = self.PE(q.shape[1])
        qkT2 = torch.matmul(q.transpose(0, 1), rpe.transpose(1, 2)).transpose(0, 1)
        qkT += qkT2 * scale

        attention = self.dropout(qkT.softmax(dim=-1))
        attention = einsum("b n m, b m d->b n d", attention, v)
        attention = rearrange(attention, "(b h) n d -> b n (h d)", h=self.num_heads)
        return attention

    def forward(self, q, k, v):
        # (q, k, v) = map(lambda x: self.PE(x), (q, k, v))
        v = self.norm(v)
        x = self._multi_head_attention(q, k, v)
        x = self.proj(x)
        return x


# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# x = torch.randn(2, 64, 128)
# model = Multi_Head_Attention(max_k=5, embed_dim=128)
# model(x, x, x).shape
# -

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        n_dim_in,
        n_dim_out,
        kernel_size=5,
        stride=1,
        padding="same",
    ):
        super().__init__()

        self.depthwise_conv = nn.Conv1d(
            n_dim_in,
            n_dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=n_dim_in,
        )
        self.pointwise_conv = nn.Conv1d(
            n_dim_out,
            n_dim_out,
            kernel_size=1,
            stride=1,
            padding=padding,
            groups=1,
        )

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-activity"]
# model = DepthwiseSeparableConv1d(32, 128)
# x = torch.randn(32, 32, 48000)
# model(x).shape
