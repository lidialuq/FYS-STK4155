# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 17:40:21 2021

@author: lidia
"""
from pathlib import Path
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from data.franke_function import get_ff_data
from data.breast_cancer import get_breastcancer
from helpers.activations import Linear, Sigmoid, ReLu, LeakyReLu, Tanh
from helpers.cost_functions import MeanSquareError, BinaryCrossEntropy, R2, Accuracy
from ffnn import FFNN

#Get dir to save plots
parent_dir = Path(__file__).parents[1]
plots_dir = join(parent_dir, 'figures')

def grid_search_ff(lmbda_vals = None, eta_vals = None, plot = True, 
                   save = False):
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 1000
    hidden_layers_struct= [50]
    
    # Load data
    X_train, X_test, z_train, z_test = get_ff_data()

    # Define learning rates and regularization parameters to try
    eta_vals = np.logspace(-5,-1,5)    
    lmbda_vals = np.logspace(-6,1,8)
    
    # Initialize arrays to save scores
    r2_all = np.zeros((len(eta_vals), len(lmbda_vals)))
    r2_all_train = np.zeros((len(eta_vals), len(lmbda_vals)))
    
    # Grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            print('Training for learning rate {} and lambda {}'.format(eta,lmbda))
            
            # Train neural net
            nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = hidden_layers_struct, 
              activation = ReLu(),   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              initialize = 'xavier',
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = lmbda, 
                     minibatch_n = 10, info = False)
            
            # Predict from train and test data
            z_train_predicted = nn.predict(X_train)
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            
            if nn.ended_in_nan == False:
                r2_all[i,j] = metrics.r2_score(z_test, z_predicted)
                r2_all_train[i,j] = metrics.r2_score(z_train, z_train_predicted)
            else: 
                r2_all[i,j] = np.nan
                r2_all_train[i,j] = np.nan
                
#            try:
#                r2_all[i,j] = metrics.r2_score(z_test, z_predicted)
#            except ValueError: 
#                r2_all[i,j] = np.nan
#                
#            try:
#                r2_all_train[i,j] = metrics.r2_score(z_train, z_train_predicted)
#            except ValueError: 
#                r2_all_train[i,j] = np.nan
                
            
    if plot:
        # Makes axis show right (:.1e if you want only scientific notation)
        lmbda_ticks = ["{}".format(i) for i in lmbda_vals]
        eta_ticks = ["{}".format(i) for i in eta_vals]
        
        # Plot test R2
        sns.set()
        sns.heatmap(r2_all, annot=True, cmap="viridis", fmt='.2f',
                    xticklabels = lmbda_ticks, yticklabels = eta_ticks)
        b, t = plt.ylim() 
        plt.ylim(b, t)
        plt.title("Test R2")
        plt.ylabel('Learning rate $\\eta$')
        plt.xlabel('Regularization coefficient $\\lambda$')
        if save: 
            name = 'test_R2_{}_{}relu.png'.format(epochs, hidden_layers_struct)
            plt.savefig(join(plots_dir, 'franke_function', name))
        plt.show()
        
        # Plot train R2
        sns.set()
        sns.heatmap(r2_all_train, annot=True, cmap="viridis", fmt='.2f',
                    xticklabels = lmbda_ticks, yticklabels = eta_ticks)
        b, t = plt.ylim() 
        plt.ylim(b, t)
        plt.title("Training R2")
        plt.ylabel('Learning rate $\\eta$')
        plt.xlabel('Regularization coefficient $\\lambda$')
        if save: 
            name = 'train_R2_{}_{}relu.png'.format(epochs, hidden_layers_struct)
            plt.savefig(join(plots_dir, 'franke_function', name))
        plt.show()
            
def grid_search_breast(lmbda_vals = None, eta_vals = None, plot = True, 
                       save = False):
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 10
    hidden_layers_struct= [100,100]
    
    # Load data
    X_train, X_test, z_train, z_test = get_breastcancer()
    
    # Define learning rates and regularization parameters to try
    eta_vals = np.logspace(-5,-1,5)    
    lmbda_vals = np.logspace(-6,1,8)
  
    # Initialize arrays to save scores
    accuracy_all = np.zeros((len(eta_vals), len(lmbda_vals)))
    accuracy_train = np.zeros((len(eta_vals), len(lmbda_vals)))
    
    # Grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbda in enumerate(lmbda_vals):
            print('Training for learning rate {} and lambda {}'.format(eta,lmbda))
            
            # Train neural net
            nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = hidden_layers_struct, 
              activation = ReLu(),   
              output_activation = Sigmoid(),
              cost_function = BinaryCrossEntropy(),
              initialize = 'xavier',
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = lmbda, 
                     minibatch_n = 10, info = False)
                        
            # Predict from train and test data
            z_train_predicted = nn.predict(X_train)
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            if nn.ended_in_nan == False:
                accuracy_all[i,j] = Accuracy()(z_test, z_predicted)
                accuracy_train[i,j] = Accuracy()(z_train, z_train_predicted)
            else:
                accuracy_all[i,j] = np.nan
                accuracy_train[i,j] = np.nan
                
    if plot:
        # Makes axis show right
        lmbda_ticks = ["{}".format(i) for i in lmbda_vals]
        eta_ticks = ["{}".format(i) for i in eta_vals]
        
        # Plot test accuracy
        sns.set()
        sns.heatmap(accuracy_all, cmap="viridis", annot=True, fmt='.2f',
                    xticklabels = lmbda_ticks, yticklabels = eta_ticks)
        plt.title("Test accuracy")
        plt.ylabel('Learning rate $\\eta$')
        plt.xlabel('Regularization coefficient $\\lambda$')
        if save: 
            name = 'test_accuracy_{}_{}.png'.format(epochs, hidden_layers_struct)
            plt.savefig(join(plots_dir, 'breast_cancer', name))
        plt.show()
        
        # Plot train accuracy
        sns.set()
        sns.heatmap(accuracy_train, annot=True, cmap="viridis", fmt='.2f',
                    xticklabels = lmbda_ticks, yticklabels = eta_ticks)
        b, t = plt.ylim() 
        plt.ylim(b, t)
        plt.title("Train accuracy")
        plt.ylabel('Learning rate $\\eta$')
        plt.xlabel('Regularization coefficient $\\lambda$')
        if save: 
            name = 'train_accuracy_{}_{}.png'.format(epochs, hidden_layers_struct)
            plt.savefig(join(plots_dir, 'breast_cancer', name))
        plt.show()
        
    
def run_ff_once(sklearn = False):

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
        
        
def run_breast_once():
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
    
if __name__ == "__main__":
    
    grid_search_ff(lmbda_vals = None, eta_vals = None, plot = True, save = True)
    #grid_search_breast(lmbda_vals = None, eta_vals = None, plot = True, save = True)

