#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:33:01 2021

@author: lidia
"""
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism

import torch
set_determinism(seed=42)


#-----------------------------------------------------------------------


# Make dataset

root = '/home/lidia/CRAI-NAS/BraTS/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'

class Dataset(monai.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, root):
        'Initialization'
        self.root = root
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        path_to_volume = path.join(self.root, 'train', ID, 'tumor.npy')
        
        X = torch.from_numpy( np.load(path_to_volume))
        X = torch.nn.functional.interpolate(X, size = (32, 256, 256))
        X = X.cpu().detach().numpy()
        X = normalize(X)
        X = torch.from_numpy(X)
        y = self.labels[ID]

        return X, y

DataLoader(data, transform=None)

#check out: https://github.com/cv-lee/BraTs/blob/master/pytorch/dataset.py 
#https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
