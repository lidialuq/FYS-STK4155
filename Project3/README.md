The aim of this project was to compare a UNet segmentation model trained with and without data augmentation. The BraTS dataset was used to train and validate the models. 

Some selected figures can be found in the figures folder, including examples of the output segmentations of the trained model. The model_metrics folder contains, for each trained model, a dictionary with the training and validation losses for each epoch (used for plotting). The data is saved with pickle. The trained models could not be saved on github due to space limitations, but can be shared upon request.

The code folder contains the following: 

* DynUnet.py: The model and everything requited to train and validate, including data preprocesing and augmentation functions.
* metrics.py: Plotting of all metrics included in the report.
* show_segmentation.py: Plotting of the segmentation examples show in the report. 
* tranform.py: Auxiliary transform for the BraTS dataset labels.


