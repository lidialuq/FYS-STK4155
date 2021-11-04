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

from data.franke_function import get_ff_data
from data.breast_cancer import get_breastcancer

'''
#%%
sklearn = False
X_train, X_test, z_train, z_test = get_ff_data()

if not sklearn:
    
    # Initialize neural network
    nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = [50,50], 
              activation = ReLu(),   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              initialize = 'xavier',
              )
    
    # Train and predict
    nn.train(X_train, z_train, epochs = 8000, eta = 0.01, lmbda = 0.01, 
             minibatch_n = 10, info = True)
    z_predicted = nn.predict(X_test)
    print(MeanSquareError()(z_predicted, z_test))

if sklearn:
    clf = MLPRegressor(solver='sgd', activation = 'identity', learning_rate_init = 0.0001,
                        hidden_layer_sizes=(), random_state=42, verbose = True)
    
    clf.fit(X_train, z_train)
    clf.predict(X_test)
'''
#%%
X_train, X_test, z_train, z_test = get_breastcancer()

# Initialize neural network
nn = FFNN(n_datapoints = X_train.shape[0],
          n_input_neurons = X_train.shape[1],
          n_output_neurons = 1, 
          hidden_layers_struct = [50], 
          activation = Sigmoid(),   
          output_activation = Sigmoid(),
          cost_function = BinaryCrossEntropy(),
          initialize = 'xavier',
          )

# Train and predict
nn.train(X_train, z_train, epochs = 100, eta = 0.0001, lmbda = 0.1, 
         minibatch_n = 10, info = True)
z_predicted = nn.predict(X_test)
print(BinaryCrossEntropy()(z_predicted, z_test))
