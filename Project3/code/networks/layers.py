# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:31:43 2021

@author: lidia
"""


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lambda_layer import LambdaLayer

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, cfg, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Choose normalization
        if cfg.norm == 'InstanceNorm2d':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        # (Convolution + Norm + ReLu) *2 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvLambda(nn.Module):
    """(LambdaLayer => [BN] => ReLU) * 2"""

    def __init__(self, cfg, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Choose normalization
        if cfg.norm == 'InstanceNorm2d':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
            
        self.double_conv = nn.Sequential(
            LambdaLayer(cfg, dim = in_channels,dim_out = mid_channels, n = 64, 
                        r = cfg.kernel_size, dim_k = 16, 
                        heads = int(mid_channels/32), dim_u = 4),       
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            LambdaLayer(cfg, dim = mid_channels,dim_out = out_channels, n = 64, r = cfg.kernel_size, dim_k = 16, heads = int(out_channels/32), dim_u = 4),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()
        if cfg.lambdaLayer:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConvLambda(cfg, in_channels, out_channels)
            )
        else: # Unet
             self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                DoubleConv(cfg, in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, cfg, in_channels, out_channels, lambdaLayer=False, bilinear=True):
        super().__init__()

        # upscale with bilinear interpolation, then double convolutions to 
        # reduce nr of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(cfg, in_channels, out_channels, in_channels // 2)
        # upscale with transpose convolution, then either double lambda
        # convolution of normal double convolution
        else:
            if lambdaLayer:
                self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, 
                                             kernel_size=2, stride=2)
                self.conv = DoubleConvLambda(cfg, in_channels, out_channels)     
            else:
                self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, 
                                             kernel_size=2, stride=2)
                self.conv = DoubleConv(cfg, in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad W and H so that x1 matches size of x2 (padding might be assimetric)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # concatinate to create Unet skip connections
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv(x)