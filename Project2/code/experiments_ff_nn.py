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
    """
    Performs the grid search of the test and train accuracy of the FFNN for the given 
    lambda inputs. Uses the Franke Fuction.
    Args:
        lmbda_vals (float):     Optional. Array of regularization parameters lambda.
                                Default None.
        eta_vals (float):       Optional. Array of learning rate parameters eta.
                                Default None.
        plot (boolean):         Optional. True if an output plot should be produced.
                                Default True.
        save (boolean):         Optional. True if an output plot should be saved. 
                                plot must be True to create the output. Default False.
    """
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 1000
    hidden_layers_struct= [50, 100, 100, 50]
    
    # Load data 
    X_train, X_test, z_train, z_test = get_ff_data()

    # Define learning rates and regularization parameters to try
    eta_vals = np.logspace(-3,-1,3)    
    lmbda_vals = np.logspace(-3,-1,3)
    
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
              seed = 42,
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
                print('r={}'.format(r2_all[i,j]))
                r2_all_train[i,j] = metrics.r2_score(z_train, z_train_predicted)
            else: 
                r2_all[i,j] = np.nan
                r2_all_train[i,j] = np.nan
                
            
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
           
    return r2_all


def grid_search_ff_activations(save = False, plot = True):
    """
    Performs the grid search of the test and train R2 of the FFNN over learning 
    rates and three activation functions, Sigmoid, ReLu and LeakyReLu. Uses the 
    Franke Fuction.
    Args:
        plot (boolean):         Optional. True if an output plot should be produced.
                                Default True.
        save (boolean):         Optional. True if an output plot should be saved. 
                                plot must be True to create the output. Default False.
    """
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 1000
    hidden_layers_struct= [50]
    
    # Define etas and activations to try
    eta_vals = np.logspace(-5,-1,5) 
    activations = (Sigmoid(), ReLu(), LeakyReLu())
    
    # Load data
    X_train, X_test, z_train, z_test = get_ff_data()
  
    # Initialize arrays to save scores
    r2_all = np.zeros((len(eta_vals), len(activations)))
    
    # Grid search

    for i, eta in enumerate(eta_vals):
        for j, activation in enumerate(activations):
            print('Training for learning rate {} with activation {}'.format(eta,
                  activation))
            
            # Train neural net
            nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = hidden_layers_struct, 
              activation = activation,   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              initialize = 'xavier',
              seed = 42,
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = 0, 
                     minibatch_n = 10, info = False)
            
            # Predict from train and test data
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            
            if nn.ended_in_nan == False:
                r2_all[i,j] = metrics.r2_score(z_test, z_predicted)
                print('r={}'.format(r2_all[i,j]))
            else: 
                r2_all[i,j] = np.nan
                      
    if plot:
        # Makes axis show right (:.1e if you want only scientific notation)
        activation_ticks = ["Sigmoid", "ReLu", "LeReLu"]
        eta_ticks = ["{}".format(i) for i in eta_vals]
        
        # Plot test R2
        sns.set()
        sns.heatmap(r2_all, annot=True, cmap="viridis", fmt='.3f',
                    xticklabels = activation_ticks, yticklabels = eta_ticks)
        b, t = plt.ylim() 
        plt.ylim(b, t)
        plt.title("Test R2")
        plt.ylabel('Learning rate $\\eta$')
        if save: 
            name = 'test_R2_{}_{}activations.png'.format(epochs, hidden_layers_struct)
            plt.savefig(join(plots_dir, 'franke_function', name))
        plt.show()
        
def grid_search_ff_hiddenlayers(save = False, plot = True):
    """
    Performs the grid search of the test and train R2 of the FFNN over learning 
    rates and different network arquitectures. Uses the Franke Fuction.
    Args:
        plot (boolean):         Optional. True if an output plot should be produced.
                                Default True.
        save (boolean):         Optional. True if an output plot should be saved. 
                                plot must be True to create the output. Default False.
    """
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 1000
    
    # Define etas and activations to try
    eta_vals = np.logspace(-3,-1,3) 
    hidden_layers_struct = ([50], [100], [50,50], [50, 100, 50], [50, 100, 100, 50])
    
    # Load data
    X_train, X_test, z_train, z_test = get_ff_data()
  
    # Initialize arrays to save scores
    r2_all = np.zeros((len(eta_vals), len(hidden_layers_struct)))
    
    # Grid search
    for i, eta in enumerate(eta_vals):
        for j, struct in enumerate(hidden_layers_struct):
            print('Training for learning rate {} with hidden layers {}'.format(eta,
                  struct))
            
            # Train neural net
            nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = struct, 
              activation = ReLu(),   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              initialize = 'xavier',
              seed = 40,
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = 0, 
                     minibatch_n = 10, info = False)
            
            # Predict from train and test data
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            
            if nn.ended_in_nan == False:
                r2_all[i,j] = metrics.r2_score(z_test, z_predicted)
                print('r={}'.format(r2_all[i,j]))
            else: 
                r2_all[i,j] = np.nan
                      
    if plot:
        # Makes axis show right (:.1e if you want only scientific notation)
        activation_ticks = ["50", "100", "50-50", "50-100-50", "50-100-100-50"]
        eta_ticks = ["{}".format(i) for i in eta_vals]
        
        # Plot test R2
        sns.set()
        sns.heatmap(r2_all, annot=True, cmap="viridis", fmt='.3f',
                    xticklabels = activation_ticks, yticklabels = eta_ticks)
        b, t = plt.ylim() 
        plt.ylim(b, t)
        plt.title("Test R2")
        plt.ylabel('Learning rate $\\eta$')
        plt.xlabel('Structure of hidden layers')

        if save: 
            name = 'test_R2_{}_hiddenlayers_lmda.png'.format(epochs)
            plt.savefig(join(plots_dir, 'franke_function', name))
        plt.show()
        
        
def ff_initialization():
    """
    Performs the grid search of the test and train R2 of the FFNN over learning 
    rates and different weight initializations (normal or xavier). Uses the Franke 
    Fuction.
    """
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 1000
    eta = 0.01
    hidden_layers_struct = [50, 100, 100, 50]
    initializations = ['normal', 'xavier']
    runs = 20
    # Load data
    X_train, X_test, z_train, z_test = get_ff_data()
  
    # Initialize arrays to save scores
    r2_all = np.zeros((len(initializations), runs))
    
    # Grid search
    for i, init in enumerate(initializations):
        for j in range(runs):
            print('Training for initialization {} run {}'.format(init,j))
            
            # Train neural net
            nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = hidden_layers_struct, 
              activation = ReLu(),   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              initialize = init,
              seed = j,
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = 0, 
                     minibatch_n = 10, info = False)
            
            # Predict from train and test data
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            
            if nn.ended_in_nan == False:
                r2_all[i,j] = metrics.r2_score(z_test, z_predicted)
            else: 
                r2_all[i,j] = np.nan
            print('r={}'.format(r2_all[i,j]))        
    
    print('Average R2 for xavier: {}'.format(np.mean(r2_all, axis=1)[0]))
    print('Average R2 for normal: {}'.format(np.mean(r2_all, axis=1)[1]))

    return r2_all


def ff_initialization_bias():
    """
    Performs the grid search of the test and train R2 of the FFNN over learning 
    rates and different bias initializations. Uses the Franke Fuction.
    """
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 1000
    eta = 0.01
    hidden_layers_struct = [50, 100, 100, 50]
    initializations = [0, 0.01]
    runs = 20
    # Load data
    X_train, X_test, z_train, z_test = get_ff_data()
  
    # Initialize arrays to save scores
    r2_all = np.zeros((len(initializations), runs))
    
    # Grid search
    for i, init in enumerate(initializations):
        for j in range(runs):
            print('Training for bias initialization {} run {}'.format(init,j))
            
            # Train neural net
            nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = hidden_layers_struct, 
              activation = ReLu(),   
              output_activation = Linear(),
              cost_function = MeanSquareError(),
              initialize = 'xavier',
              initialize_bias = init,
              seed = j,
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = 0, 
                     minibatch_n = 10, info = False)
            
            # Predict from train and test data
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            
            if nn.ended_in_nan == False:
                r2_all[i,j] = metrics.r2_score(z_test, z_predicted)
            else: 
                r2_all[i,j] = np.nan
            print('r={}'.format(r2_all[i,j]))        
    
    print('Average R2 for bias init {}: {}'.format(initializations[0], np.mean(r2_all, axis=1)[0]))
    print('Average R2 for bias init {}: {}'.format(initializations[1], np.mean(r2_all, axis=1)[1]))

    return r2_all
        
def train_once(sklearn = False):
    """
    Performs a single estimation of FFNN for the predefined inputs using sklearn or handmade solution.
    Uses the Franke Function.
    Args:
        sklearn (boolean):      Optional. True if sklear should be used. Default False.
    """

    X_train, X_test, z_train, z_test = get_ff_data()
    
    if not sklearn:
        
        # Initialize neural network
        nn = FFNN(n_datapoints = X_train.shape[0],
                  n_input_neurons = X_train.shape[1],
                  n_output_neurons = 1, 
                  hidden_layers_struct = [50,100,100,50], 
                  activation = ReLu(),   
                  output_activation = Linear(),
                  cost_function = MeanSquareError(),
                  initialize = 'xavier',
                  seed = 42,
                  )
        
        # Train and predict
        nn.train(X_train, z_train, epochs = 1000, eta = 0.01, lmbda = 0, 
                 minibatch_n = 10, info = False)
        z_predicted = nn.predict(X_test)
        print('R2 score = {}'.format(metrics.r2_score(z_test, z_predicted)))
    
    if sklearn:
        clf = MLPRegressor(solver='sgd',
                           activation = 'relu', 
                           learning_rate_init = 0.01,
                           hidden_layer_sizes=(50,100,100,50), 
                           alpha = 0,
                           batch_size = 10,
                           random_state=42, 
                           verbose = False)
        
        clf.fit(X_train, z_train)
        z_predicted = clf.predict(X_test)
        print('R2 score = {}'.format(metrics.r2_score(z_test, z_predicted)))
        
    
if __name__ == "__main__":
    # Functions are run from here for analysis. For example:
    # grid_search_ff(lmbda_vals = None, eta_vals = None, plot = True, save = True)
