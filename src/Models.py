
# Importing the necessary modules
import os                   # Operating System functions (read/write files, create directories etc.)
from pathlib import Path    # Object-oriented filesystem paths
import imageio              # Reading/Writing a wide range of image data
import matplotlib.pyplot as plt  # Plotting library
import numpy as np          # Numerical computations library
from PIL import Image       # Python Imaging Library
import torch                # PyTorch machine learning library
import torch.nn as nn           # Neural Networks module in PyTorch
from torchvision import transforms  # Transformations for image data


class UNet(nn.Module):
    """U-Net implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    # _conv_block and _upsampler are just helper functions to
    # construct the model.
    # encapsulating them like so also makes it easy to re-use
    # the model implementation with different architecture elements

    # Convolutional block for single layer of the decoder / encoder
    # we apply two 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    # upsampling via transposed 2d convolutions 
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def __init__(self, in_channels=1, out_channels=1, depth=4, feat=16, final_activation=None):
        super().__init__()

        assert depth < 10, "Max supported depth is 9"

        # the depth (= number of encoder / decoder levels) is
        # hard-coded to 4
        self.depth = depth
        self.feat = feat

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(
                final_activation, nn.Module
            ), "Activation must be torch module"

        # all lists of conv layers (or other nn.Modules with parameters) must be wraped
        # itnto a nn.ModuleList

        # modules of the encoder path
        self.encoder = nn.ModuleList(
            [
                self._conv_block(in_channels, feat),
                self._conv_block(feat, feat*2),
                self._conv_block(feat*2, feat*4),
                self._conv_block(feat*4, feat*8),
                self._conv_block(feat*8, feat*16),
                self._conv_block(feat*16, feat*32),
                self._conv_block(feat*32, feat*64),
                self._conv_block(feat*64, feat*128),
                self._conv_block(feat*128, feat*256),
            ][:depth]
        )
        # the base convolution block
        if depth >= 1:
            self.base = self._conv_block(feat * (2 ** (depth-1)), feat* 2 ** (depth))
        else:
            self.base = self._conv_block(1, feat * (2 ** depth))
            
        # modules of the decoder path
        self.decoder = nn.ModuleList(
            [
                self._conv_block(feat*512, feat*256),
                self._conv_block(feat*256, feat*128),
                self._conv_block(feat*128, feat*64),
                self._conv_block(feat*64, feat*32),
                self._conv_block(feat*32, feat*16),
                self._conv_block(feat*16, feat*8),
                self._conv_block(feat*8, feat*4),
                self._conv_block(feat*4, feat*2),
                self._conv_block(feat*2, feat),
            ][-depth:]
        )

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool2d(2) for _ in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList(
            [
                self._upsampler(feat*512, feat*256),
                self._upsampler(feat*256, feat*128),
                self._upsampler(feat*128, feat*64),
                self._upsampler(feat*64, feat*32),
                self._upsampler(feat*32, feat*16),
                self._upsampler(feat*16, feat*8),
                self._upsampler(feat*8, feat*4),
                self._upsampler(feat*4, feat*2),
                self._upsampler(feat*2, feat),
            ][-depth:]
        )
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv2d(self.feat, out_channels, 1)
        self.activation = final_activation

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](torch.cat((x, encoder_out[level]), dim=1))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    


