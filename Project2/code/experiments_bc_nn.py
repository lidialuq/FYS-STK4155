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
from sklearn.neural_network import MLPRegressor, MLPClassifier

from data.franke_function import get_ff_data
from data.breast_cancer import get_breastcancer
from helpers.activations import Linear, Sigmoid, ReLu, LeakyReLu, Tanh
from helpers.cost_functions import MeanSquareError, BinaryCrossEntropy, R2, Accuracy
from ffnn import FFNN

#Get dir to save plots
parent_dir = Path(__file__).parents[1]
plots_dir = join(parent_dir, 'figures')


def grid_search_breast(plot = True, save = False): 
    """
    Performs the grid search of the test and train accuracy of the FFNN for the given 
    lambda inputs. Uses the Wisconsin Breast Cancer Data.
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
    hidden_layers_struct= [50]
    
    # Load data
    X_train, X_test, z_train, z_test = get_breastcancer()
    
    # Define learning rates and regularization parameters to try
    eta_vals = np.logspace(-5,-1,5)    
    lmbda_vals = np.logspace(-5,0,6)
  
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
              seed = 42,
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = lmbda, 
                     minibatch_n = 100, info = False)
                        
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
        sns.heatmap(accuracy_all, cmap="viridis", annot=True, fmt='.3f',
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


def grid_search_breast_hiddenlayers(save = False, plot = True):
    """
    Performs the grid search of the test and train accuracy of the FFNN for the 
    different learning rates and network structures. Uses the Wisconsin Breast 
    Cancer Data.
    Args:
        plot (boolean):         Optional. True if an output plot should be produced.
                                Default True.
        save (boolean):         Optional. True if an output plot should be saved. 
                                plot must be True to create the output. Default False.
    """
    
    # Set parameters, this will define neural net and filename to save plot
    epochs = 100
    
    # Define etas and activations to try
    eta_vals = np.logspace(-3,-1,3) 
    hidden_layers_struct = ([50], [100], [500,500])
    
    # Load data
    X_train, X_test, z_train, z_test = get_breastcancer()
  
    # Initialize arrays to save scores
    accuracy_all = np.zeros((len(eta_vals), len(hidden_layers_struct)))
    
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
              output_activation = Sigmoid(),
              cost_function = BinaryCrossEntropy(),
              initialize = 'xavier',
              seed = 42,
              )
            nn.train(X_train, z_train, epochs = epochs, eta = eta, lmbda = 0.1, 
                     minibatch_n = 10, info = False)
            
            # Predict from train and test data
            z_predicted = nn.predict(X_test) 
            
            # Calculate scores. If one of the inputs to calculate the score 
            # is nan, set score to nan too
            if nn.ended_in_nan == False:
                accuracy_all[i,j] = Accuracy()(z_test, z_predicted)
            else:
                accuracy_all[i,j] = np.nan
                      
    if plot:
        # Makes axis show right (:.1e if you want only scientific notation)
        activation_ticks = ["50", "100", "500-500"]
        eta_ticks = ["{}".format(i) for i in eta_vals]
        
        # Plot test R2
        sns.set()
        sns.heatmap(accuracy_all, annot=True, cmap="viridis", fmt='.3f',
                    xticklabels = activation_ticks, yticklabels = eta_ticks)
        b, t = plt.ylim() 
        plt.ylim(b, t)
        plt.title("Test accuracy")
        plt.ylabel('Learning rate $\\eta$')
        plt.xlabel('Structure of hidden layers')

        if save: 
            name = 'accuracy_{}_hiddenlayers.png'.format(epochs)
            plt.savefig(join(plots_dir, 'breast_cancer', name))
        plt.show()
        
def confusion_matrix():
    """
    Get confusion matrix using the best preforming parameters as found during the
    grid search.
    """
    X_train, X_test, z_train, z_test = get_breastcancer()
    # Initialize neural network
    nn = FFNN(n_datapoints = X_train.shape[0],
              n_input_neurons = X_train.shape[1],
              n_output_neurons = 1, 
              hidden_layers_struct = [50], 
              activation = ReLu(),   
              output_activation = Sigmoid(),
              cost_function = BinaryCrossEntropy(),
              initialize = 'xavier',
              )
    
    # Train and predict
    nn.train(X_train, z_train, epochs = 100, eta = 0.01, lmbda = 0.1, 
             minibatch_n = 10, info = False)
    z_predicted = nn.predict(X_test)
    # Apply threshold to output and get confusion matrix
    threshold = 0.8
    print('Accuracy = {}'.format(Accuracy()(z_test, z_predicted, threshold)))
    z_predicted = np.where(z_predicted >= threshold, 1, 0)
    conf = metrics.confusion_matrix(z_test, z_predicted)
    print(conf)
    
def average_accuracy():
    """
    Get average accuracy by running network 100 times with the best parameters.
    """
    tries = 100
    accuracy = np.ones(tries)
    conf = np.ones((2,2))
    X_train, X_test, z_train, z_test = get_breastcancer()


    for i in range(tries): 

        # Initialize neural network
        nn = FFNN(n_datapoints = X_train.shape[0],
                  n_input_neurons = X_train.shape[1],
                  n_output_neurons = 1, 
                  hidden_layers_struct = [50], 
                  activation = ReLu(),   
                  output_activation = Sigmoid(),
                  cost_function = BinaryCrossEntropy(),
                  initialize = 'xavier',
                  seed = i
                  )
        
        # Train and predict
        nn.train(X_train, z_train, epochs = 100, eta = 0.01, lmbda = 0.1, 
                 minibatch_n = 10, info = False)
        z_predicted = nn.predict(X_test)
        # Apply threshold to output and get confusion matrix
        threshold = 0.8
        accuracy[i] = Accuracy()(z_test, z_predicted, threshold)
        z_predicted = np.where(z_predicted >= threshold, 1, 0)
        conf += metrics.confusion_matrix(z_test, z_predicted)
    
    np.set_printoptions(suppress=True)
    print(conf)
    print('Mean accuracy = {}'.format(np.mean(accuracy)))
        
        
def train_breast_once(sklearn = False):
    """
    Performs a single estimation of FFNN for the predefined inputs using sklearn or handmade solution.
    Uses the Wisconsin Breast Cancer Data.
    Args:
        sklearn (boolean):      Optional. True if sklear should be used. Default False.
    """

    X_train, X_test, z_train, z_test = get_breastcancer()
    
    if not sklearn:
        
        # Initialize neural network
        nn = FFNN(n_datapoints = X_train.shape[0],
                  n_input_neurons = X_train.shape[1],
                  n_output_neurons = 1, 
                  hidden_layers_struct = [50], 
                  activation = ReLu(),   
                  output_activation = Sigmoid(),
                  cost_function = BinaryCrossEntropy(),
                  initialize = 'xavier',
                  seed = 42,
                  )
        
        # Train and predict
        nn.train(X_train, z_train, epochs = 100, eta = 0.01, lmbda = 0.1, 
                 minibatch_n = 10, info = False)
        z_predicted = nn.predict(X_test)
        print('Accuracy = {}'.format(Accuracy()(z_test, z_predicted)))
    
    if sklearn:
        clf = MLPClassifier(solver='sgd',
                           activation = 'relu', 
                           learning_rate_init = 0.01,
                           hidden_layer_sizes=(50), 
                           alpha = 0.1,
                           batch_size = 10,
                           random_state=42, 
                           verbose = False)
        
        clf.fit(X_train, z_train)
        z_predicted = clf.predict(X_test)
        print('Accuracy = {}'.format(metrics.accuracy_score(z_test, z_predicted)))
        
    
if __name__ == "__main__":
    # Functions are run from here for analysis. For example:
    #grid_search_breast(plot = True, save = False)
