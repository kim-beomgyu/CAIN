import math
import numpy as np

import torch
import torch.nn as nn

from .common import *


class Encoder(nn.Module):
    def __init__(self, in_channels=1, depth=3):
        super(Encoder, self).__init__()

        # Shuffle pixels to expand in channel dimension
        # shuffler_list = [PixelShuffle(0.5) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(1 / 2**depth)

        relu = nn.LeakyReLU(0.2, True)
        
        # FF_RCAN or FF_Resblocks
        self.interpolate = Interpolation_pred(5, 12, in_channels * (4**depth), act=relu)

    def forward(self, x_list):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats = [self.shuffler(x) for x in x_list]

        feats = self.interpolate(feats)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        # shuffler_list = [PixelShuffle(2) for i in range(depth)]
        # self.shuffler = nn.Sequential(*shuffler_list)
        self.shuffler = PixelShuffle(2**depth)

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN_Pred(nn.Module):
    def __init__(self, depth=3):
        super(CAIN, self).__init__()
        
        self.encoder = Encoder(in_channels=1, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, x):

        # x1, m1 = sub_mean(x1)
        # x2, m2 = sub_mean(x2)
        x_list = []
        m_list = []
        for x_i in x:
            x_i, m_i = sub_mean(x_i)
            x_list.append(x_i)
            m_list.append(m_i)


        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x_list[0])
            x_list = [paddingInput(x_i) for x_i in x_list]

            # paddingInput, paddingOutput = InOutPaddings(x1)
            # x1 = paddingInput(x1)
            # x2 = paddingInput(x2)

        feats = self.encoder(x_list)
        out = self.decoder(feats)

        if not self.training:
            out = paddingOutput(out)

        mi = sum(m_list) / len(m_list)
        out += mi

        return out, feats
