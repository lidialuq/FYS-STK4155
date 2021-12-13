#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:34:15 2021

@author: lidia
"""
import pickle
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
from monai.networks.nets import DynUNet
from monai.data import DataLoader
from monai.apps import DecathlonDataset
from monai.transforms import (
    Compose,
    LoadImaged, 
    EnsureChannelFirstd, 
    Orientationd, 
    Resized,
    CropForegroundd,
    SpatialPadd,
    NormalizeIntensityd, 
    EnsureTyped, 
    EnsureType,
    Activations,
    AsDiscrete,
)
from transform import ConvertToMultiChannelBasedOnBratsClassesd

data_dir = '/home/lidia/CRAI-NAS/BraTS/BraTs_2016-17'
model_folder = "/home/lidia/CRAI-NAS/lidfer/Segmentering/saved_models"
save_figures = "/home/lidia/Projects/fys-stk4155/Project3/figures"
best_model_folder = os.path.join(model_folder, 'DynUnet_noaug_3deepsup_5e4')

##############################################################################
# Plot three examples of segmention from best model
##############################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_transform = Compose([       
        # Load images, set first dim to channel, resize and normalize
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(128,128,128)),
        Resized(keys=["image", "label"], spatial_size=(128,128,128), align_corners=[False,None],
                mode=["trilinear", "nearest"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        
        # Convert to tensor
        EnsureTyped(keys=["image", "label"], data_type='tensor'),
    ])

post_trans = Compose([
    EnsureType(data_type='tensor'), 
    Activations(sigmoid=True), 
    AsDiscrete(threshold_values=0.5),
    ])

val_ds = DecathlonDataset(
    root_dir=data_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)

val_loader = DataLoader(val_ds, batch_size=3, shuffle=False, num_workers=4)


model = DynUNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    kernel_size= [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides= [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size= [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    norm_name="instance",
    deep_supervision=True,
    deep_supr_num=2,
).to(device)

model.load_state_dict(
torch.load(os.path.join(best_model_folder, "best_metric_model.pth"),  map_location=device)
)
model.eval()
figure_nr = 6
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    
    # Unsqueeze removes batch dimension
    input_volume = val_ds[figure_nr]["image"].unsqueeze(0).to(device)


    output = post_trans(model(input_volume)[0])
    output_numpy = output.detach().numpy()
    input_numpy = input_volume[0].detach().numpy()
    label = val_ds[figure_nr]["label"].detach().cpu()
    label_numpy = label.detach().numpy()
    
    
    # Plot all channels of input to model 
    fig, ax = plt.subplots(1, 4, figsize=(10,17), squeeze=True)
    for i in range(4):
        ax[i].imshow(input_numpy[i,:,:,70], cmap='gray')
        ax[i].set_xticks([]) 
        ax[i].set_yticks([]) 
    ax[0].set_title('Flair')
    ax[1].set_title('T1')
    ax[2].set_title('T1c')
    ax[3].set_title('T2')

    plt.tight_layout()
    plt.savefig(os.path.join(save_figures, f'inputs_pat{figure_nr}'))
    plt.show()
    
    # Plot comparison of model output and label with flair image as backgroud
    red_cmap = plt.cm.Reds
    red_cmap.set_under(color="white", alpha="0")
    blue_cmap = plt.cm.Blues
    blue_cmap.set_under(color="white", alpha="0")    
    
    fig, ax = plt.subplots(3, 2, figsize=(4.5,6.75), squeeze=True)

    slice_nr = 65
    for i in range(3):
        ax[i,0].imshow(output_numpy[i,:,:,slice_nr], cmap=red_cmap)
        ax[i,0].imshow(input_numpy[0,:,:,slice_nr], cmap="gray", alpha=0.6)
        ax[i,0].set_xticks([]) 
        ax[i,0].set_yticks([]) 
    for i in range(3):
        ax[i,1].imshow(label_numpy[i,:,:,slice_nr], cmap=blue_cmap)
        ax[i,1].imshow(input_numpy[0,:,:,slice_nr], cmap="gray", alpha=0.6)
        ax[i,1].set_xticks([]) 
        ax[i,1].set_yticks([]) 
        
    ax[0,0].set_title('Model output')
    ax[0,1].set_title('Label')
    ax[0,0].set_ylabel('Tumor Core')
    ax[1,0].set_ylabel('Whole Tumor')
    ax[2,0].set_ylabel('Enhanced Tumor')

    plt.tight_layout()
    plt.savefig(os.path.join(save_figures, f'segmentation_pat{figure_nr}'))
    plt.show()
