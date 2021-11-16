# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:13:48 2021

@author: lidia

Code shamelessly copied (and lightly adapted) from https://medium.com/@ya.aman.ay/
a-simple-machine-learning-model-to-predict-breast-cancer-in-python-1919d7ad04c6

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def get_breastcancer():
    
    # Load dataset and get X and y matrixes
    dataset = load_breast_cancer()
    #print(dataset.DESCR) # Print the data set description
    df = pd.DataFrame(dataset.data, columns = [dataset.feature_names])
    df_target = pd.Series( data = dataset.target, index = df.index)
    X = df.iloc[0:, 0:30].to_numpy()
    y = df_target.to_numpy()
    
    # Split data
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    # Calculate mean and std of features  
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    
    # Normalize data
    X_train = ( X_train - X_train_mean) / X_train_std
    X_test = ( X_test - X_train_mean) / X_train_std
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_breastcancer()
    print('X_train.shape = {}'.format(X_train.shape))
    print('X_test.shape = {}'.format(X_test.shape))
    print('y_train.shape = {}'.format(y_train.shape))
    print('y_test.shape = {}'.format(y_test.shape))
