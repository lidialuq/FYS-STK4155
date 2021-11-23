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
from monai.data import DataLoader#, decollate_batch
#from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms import (
    CropForegroundd,
    CropForeground,
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    #Invertd,
    LoadImaged,
    LoadImage,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    #EnsureChannelFirstd,
    #EnsureTyped,
    #EnsureType,
)
from monai.utils import set_determinism
import sys
from os.path import dirname, abspath



###############################################################################
# Configuration
###############################################################################

set_determinism(seed=42)
parent_dir = dirname(dirname(abspath(__file__)))
root_dir = os.path.join(parent_dir, 'data', 'BraTs_decathlon', 'Task01_BrainTumour')

print(root_dir)

train_transform = Compose(
    [
     LoadImaged(keys=["image", "label"]),
     #EnsureChannelFirst(),
     Orientationd(keys=["image", "label"], axcodes="RAS"),
     CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

#-----------------------------------------------------------------------
# Import dataset
# here we don't cache any data in case out of memory issue
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    #transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)


###############################################################################
# Visualize
###############################################################################

# pick one image from DecathlonDataset to visualize and check the 4 channels
print(f"image shape: {val_ds[2]['image'].shape}")
plt.figure("image", (24, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(train_ds[2]["image"][:, :, 60, i], cmap="gray")
plt.show()
# also visualize the 3 channels label corresponding to this image
print(f"label shape: {val_ds[2]['label'].shape}")
plt.figure("label", (18, 6))
plt.title(f"label channel {i}")
plt.imshow(train_ds[2]["label"][:, :, 60])
plt.show()

#check out: https://github.com/cv-lee/BraTs/blob/master/pytorch/dataset.py 
#https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
