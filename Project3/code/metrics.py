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

model_folder = "/home/lidia/CRAI-NAS/lidfer/Segmentering/saved_models"
save_figures = "/home/lidia/Projects/fys-stk4155/Project3/figures"

##############################################################################
# Print summary of metrics for all models 
##############################################################################

for model_dir in glob(os.path.join(model_folder, '*')):
    # read dictionary with pickle 
    with open(os.path.join(model_dir, "metrics.pth"), 'rb') as f:
        metrics = pickle.load(f)
        
    model = os.path.basename(model_dir)
    max_value = max(metrics["dice"])
    max_index = metrics["dice"].index(max_value)
    
    print(f'{model}: Max Dice {max_value:.3f} at epoch {max_index+1}')
    
    
##############################################################################
# Plot individual dice losses for all models 
##############################################################################

# read dictionary
# metrics_dict = {'loss': epoch_loss_values,
#                 'deep losses': deep_epoch_loss_values,
#                 'dice': metric_values,
#                 'dice wt': metric_values_wt,
#                 'dice tc': metric_values_tc,
#                 'dice et': metric_values_et,
#                 } 

for model_dir in glob(os.path.join(model_folder, '*')):
    # read dictionary with pickle 
    with open(os.path.join(model_dir, "metrics.pth"), 'rb') as f:
        metrics = pickle.load(f)
        
    # plot dice scores
    #nr_epochs = len(metrics["dice"])
    nr_epochs = 200
    x = list(range(1, nr_epochs+1))
    model = os.path.basename(model_dir)
    
    fig, ax = plt.subplots(figsize=[5,7])
    #plt.title("Mean Dice-scores" + model)
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
# Compare dice losses for best preforming learning rate with and without
# augmentation 
##############################################################################

best_aug = os.path.join(model_folder, 'DynUnet_3deepsup_5e4')
best_noaug = os.path.join(model_folder, 'DynUnet_noaug_3deepsup_5e4')

with open(os.path.join(best_aug, "metrics.pth"), 'rb') as f:
    metrics_aug = pickle.load(f)
    
with open(os.path.join(best_noaug, "metrics.pth"), 'rb') as f:
    metrics_noaug = pickle.load(f)
    
nr_epochs = 200
x = list(range(1, nr_epochs+1))

fig, ax = plt.subplots()#figsize=[5,7])
#plt.title("Dice score" + model)
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

# DynUnet_noaug_2deepsup_1e3: Max Dice 0.777 at epoch 133
# DynUnet_noaug_3deepsup_5e4: Max Dice 0.780 at epoch 93
# DynUnet_3deepsup_5e4: Max Dice 0.767 at epoch 139
# DynUnet_2deepsup_1e3: Max Dice 0.760 at epoch 138


    
