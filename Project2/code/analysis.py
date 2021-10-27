# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:30:13 2021

@author: lidia
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from helpers.activations import Linear, Sigmoid, ReLu, LeakyReLu, Tanh
from helpers.cost_functions import MeanSquareError, BinaryCrossEntropy
from ffnn import FFNN

from data.franke_function import FrankeFunction

sklearn = False

# create data
ff = FrankeFunction(axis_n = 10, noise_var = 0, plot = False)
X = ff.design_matrix(ff.x, ff.y, degree = 5)

# split data and normalize by removing mean
X_train, X_test, z_train, z_test = \
    train_test_split(X, ff.z, test_size = 0.3, shuffle = True)
#for i in range(X_train.shape[1]):
#    X_train[:,i] = X_train[:,i] - np.mean(X_train[:,i])
#    X_test[:,i]  = X_test[:,i] - np.mean(X_test[:,i])
#z_train = z_train - np.mean(z_train)
#z_test = z_test - np.mean(z_test)

z_train = np.expand_dims(z_train, 1)
z_test = np.expand_dims(z_test, 1)

if not sklearn:
    
    # Initialize neural network
    nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = [], 
              activation = Linear(),   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              )
    
    # Train and predict
    nn.train(X_train, z_train, epochs = 10, eta = 0.01, info = True)
    z_predicted = nn.predict(X_test)
    print(MeanSquareError()(z_predicted, z_test))

if sklearn:
    clf = MLPRegressor(solver='sgd', activation = 'identity', learning_rate_init = 0.0001,
                        hidden_layer_sizes=(), random_state=42, verbose = True)
    
    clf.fit(X_train, z_train)
    clf.predict(X_test)
