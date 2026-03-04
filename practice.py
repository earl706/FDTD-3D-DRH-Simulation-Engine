from __future__ import annotations

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from scipy import ndimage

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class Conv3D_Block(nn.Module):
    def __init__(
        self,
        inp_feat,
        out_feat,
        kernel=3,
        stride=1,
        padding=1,
        residual=None,
        dropout_rate=0.2,
    ):
        super(Conv3D_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                inp_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_feat,
                out_feat,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )

        self.residual = residual
        if self.residual == "conv" and inp_feat != out_feat:
            self.residual_upsampler = nn.Conv3d(
                inp_feat, out_feat, kernel_size=1, bias=False
            )
        else:
            self.residual_upsampler = None

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual == "conv":
            if self.residual_upsampler is not None:
                res = self.residual_upsampler(res)
            return out + res
        return out
