#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:12:31 2021

@author: lidia
"""
import pickle
import os
from glob import glob
import matplotlib.pyplot as plt

"""
GIven a folder with saved metrics of models:
    1) Find and print the scores for the epoch with the best test accuracy
    2) Plot class specific test scores as a funcion of training epoch for all 
       models
    3) Plot overall test scores the best model with and without augmentation
To use, set the path to the folder with the saved models, and the folder 
where the resulting figures will be saved to.
"""

##############################################################################
# Set path to models and path where figures will be saved
##############################################################################

model_folder = "/home/lidia/CRAI-NAS/lidfer/Segmentering/saved_models"
save_figures = "/home/lidia/Projects/fys-stk4155/Project3/figures"

##############################################################################
# Print summary of metrics for all models, including class and overall Dice
##############################################################################

for model_dir in glob(os.path.join(model_folder, '*')):
    # read dictionary with pickle 
    with open(os.path.join(model_dir, "metrics.pth"), 'rb') as f:
        metrics = pickle.load(f)
        
    model = os.path.basename(model_dir)
    max_value = max(metrics["dice"])
    max_index = metrics["dice"].index(max_value)
    max_wt = metrics["dice wt"][max_index]
    max_tc = metrics["dice tc"][max_index]
    max_et = metrics["dice et"][max_index]
    print(f'{model}: Max Dice {max_value:.3f} at epoch {max_index+1}. \
          WT={max_wt:.3f}, TC={max_tc:.3f}, ET={max_et:.3f}')
    
    
##############################################################################
# Plot individual overall dice scores for all models 
##############################################################################

for model_dir in glob(os.path.join(model_folder, '*')):
    
    # read dictionary containing scores with pickle 
    with open(os.path.join(model_dir, "metrics.pth"), 'rb') as f:
        metrics = pickle.load(f)
        
    # plot dice scores as a function of epoch
    nr_epochs = 200
    x = list(range(1, nr_epochs+1))
    model = os.path.basename(model_dir)
    
    fig, ax = plt.subplots(figsize=[5,7])
    y_wt = metrics["dice wt"][:nr_epochs]
    y_tc = metrics["dice tc"][:nr_epochs]
    y_et = metrics["dice et"][:nr_epochs]
    plt.xlabel("Epoch number")
    plt.ylabel("Dice score")
    WT = ax.plot(x, y_wt, color="c", label='WT')
    TC = ax.plot(x, y_tc, color="m", label='TC')
    ET = ax.plot(x, y_et, color="y", label='ET')
    ax.legend()
    ax.set_ylim([0.3, 0.95])
    plt.savefig(os.path.join(save_figures, f'dicescores_{model}'))
    plt.show()

##############################################################################
# Compare dice scores for best preforming learning rate with and without
# augmentation 
##############################################################################

# Load saved metrics for model with and without augmentation
best_aug = os.path.join(model_folder, 'DynUnet_3deepsup_5e4')
best_noaug = os.path.join(model_folder, 'DynUnet_noaug_3deepsup_5e4')

with open(os.path.join(best_aug, "metrics.pth"), 'rb') as f:
    metrics_aug = pickle.load(f)
    
with open(os.path.join(best_noaug, "metrics.pth"), 'rb') as f:
    metrics_noaug = pickle.load(f)
    
# Plot
nr_epochs = 200
x = list(range(1, nr_epochs+1))
fig, ax = plt.subplots()
y_aug = metrics_aug["dice"][:nr_epochs]
y_noaug = metrics_noaug["dice"][:nr_epochs]
plt.xlabel("Epoch number")
plt.ylabel("Dice score")
noaug = ax.plot(x, y_noaug, color="m", label='No augmentation')
aug = ax.plot(x, y_aug, color="c", label='Augmentation')
ax.legend()
plt.savefig(os.path.join(save_figures, 'Dice_aug-vs-noaug'))
plt.show()

##############################################################################
# Training times
#
# DynUnet_3deepsup_5e4: 21h 52m
# DynUnet_noaug_3deepsup_5e4: 17h 11m
# DynUnet_noaug_2deepsup_1e3: 19h 40m
# DynUnet_2deepsup_1e3: 23h 38m



    
