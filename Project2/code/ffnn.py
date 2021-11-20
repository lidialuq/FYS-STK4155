# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:19:37 2021
@author: lidialuq
"""
 
import math
import numpy as np
from sklearn.utils import resample, shuffle

class FFNN:
"""
A flexible Feed Forward Neural Network class that can be used for both
regression and classification problems, with a choice of activation functions
and cost functions.

Attributes:
    n_datapoins (int):          Number of datapoints. Default None.
    n_input_neurons (int):      Amount of input neurons
    n_output_neurons (int):     Amount of output neurons
    hidden_layers_struct (list):List of hidden layers.
    activation (function):      Activation function on the hidden layers.
                                Example: Linear(), Sigmoid(), ReLu(), LeakyReLu()
    output_activation (function): Activation function on the output layers.
                                Example: Linear(), Sigmoid(), ReLu(), LeakyReLu()
    cost_function (function):   Applied cost function. Example: MeanSquareError(), 
                                BinaryCrossEntropy()
    initialize (str):           'normal' or 'xavier', defines inintial weights

Methods:
    predict():
        '''
        Use _feed_forward() method to make predictions for X_test_matrix
        '''
    train():
        '''
        Train the network using stochastic gradient descent with mini-batches
        of size minibatch_n. Thelearning rate is eta and the network is trained
        for n epochs. X_matrix is the training data, and y_matrix the target 
        data.
        '''
"""

    def __init__(self, 
             n_datapoints: int,
             n_input_neurons: int,
             n_output_neurons: int, 
             hidden_layers_struct: list, 
             activation,   
             output_activation,
             cost_function,
             initialize: str = 'normal',
             ):
        
        # Values that define structure of neural network
        self.n_datapoints = None
        self.n_output_neurons = n_output_neurons
        self.n_input_neurons = n_input_neurons
        self.hidden_layers_struct = hidden_layers_struct
        self.n_hidden_layers = len(hidden_layers_struct)
        self.activation = activation
        self.output_activation = output_activation
        self.cost_function = cost_function
        self.initialize = initialize # normal or xavier
        
        # Create starting weights and biases
        self.w = self._initialize_weights()
        self.b = self._initialize_biases()
        
        # Did the training end in nan? (due to exploding gradients)
        self.ended_in_nan = False


    def _initialize_weights(self):
        '''
        Initialize weights with either a normal distribution (mean 0, var 1)
        or the normalized xavier distribution (helps with exploding gradients)
        Returns a list containing 2d nd.nparrays with the weights for each layer

        Returns:
            w (list):         List of weights for each layer
        '''

        def xavier_initialization(n, m): 
            array = np.random.uniform( -(np.sqrt(6)/np.sqrt(n + m)), 
                                      np.sqrt(6)/np.sqrt(n + m), 
                                      (n,m) )
            return array
        
        w = []
        
        if self.initialize == 'normal': ini_func = np.random.randn
        elif self.initialize == 'xavier': ini_func = xavier_initialization
        
        # if no hidden layer
        if self.n_hidden_layers == 0:
            w.append( ini_func(self.n_input_neurons, 
                                      self.n_output_neurons))
        else:
            # Weights of input neurons
            w.append( ini_func(self.n_input_neurons, 
                                      self.hidden_layers_struct[0]))
            
            # Weights of neurons between hidden layers (if only 1 hidden layer, 
            # nothing is appended)
            for i in range(0, self.n_hidden_layers-1):
                w.append( ini_func(self.hidden_layers_struct[i], 
                                             self.hidden_layers_struct[i+1]))
            # Weights of last hidden layer
            w.append( ini_func(self.hidden_layers_struct[-1], 
                                      self.n_output_neurons))
        
        return w
    
    def _initialize_biases(self):
        '''
        Initialize biases to zero
        Returns a list containing 1d np.ndarrays with the biases for each layer 

        Returns:
            w (list):         List of biases for each layer
        '''

        b = []
        # biases for hidden layers
        for i in range(self.n_hidden_layers):
            b.append( np.zeros(self.hidden_layers_struct[i]))
        # biases for output layer
        b.append(np.zeros(self.n_output_neurons))
        
        return b
    
    def _initialize_activations(self):
        '''
        Initializes activations. First layer (first element in list) is set 
        equal to the input data, all other layers initialized to null

        Returns:
            a (list):         List of activations for each layer
        '''

        a = []
        # activations for hidden layers
        for i in range(self.n_hidden_layers):
            a.append( np.zeros((self.n_datapoints, self.hidden_layers_struct[i])))
        # activations for output layer
        a.append( np.zeros((self.n_datapoints, self.n_output_neurons)))
        return a
        
    def _feed_forward(self):
        """
        Calculate activations for all layers.

        Returns:
            a(float):       Value of the output layer activation function           
        """
        self.z = self._initialize_activations()
        self.a = self._initialize_activations()
        
        # Network without hidden layers
        if self.n_hidden_layers == 0:
            self.z[0] = self.X_matrix @ self.w[0] + self.b[0]
            self.a[0] = self.activation( self.z[0] )

        # Network with hidden layers
        else: 
            # Calculate activations for hidden layers. First hidden layer activation
            # is calculated from the input data
            for i in range(self.n_hidden_layers):
                if i == 0:
                    self.z[i] = self.X_matrix @ self.w[i] + self.b[i]
                    self.a[i] = self.activation( self.z[i] )
                else:
                    self.z[i] = self.a[i-1] @ self.w[i] + self.b[i] 
                    self.a[i] = self.activation( self.z[i] ) 
                    
    		# Calculate activations for output layer (the network outputs)
            self.z[-1] = self.a[-2] @ self.w[-1] + self.b[-1] 
            self.a[-1] = self.output_activation( self.z[-1] ) 

        return self.a[-1] # maybe not needed


    def _backpropagate(self, eta, lmbda):
        '''
        Update weights and gradients for each layer by calculating the 
        gradients of the cost function. The gradient of the cost function 
        w.r.t y_predicted (here called error) is used to calculate the 
        gradients w.r.t the weights and the biases.
        '''
        
        # For each layer, uncluding input layer, calculate 
        for i in range(self.n_hidden_layers, -1, -1):
            
            # if last layer, calculate error for output layer. Otherwise, 
            # caluclate error for hidden layer
            if i == self.n_hidden_layers:
                delta = self.cost_function.gradient(self.y_matrix, self.a[i]) * \
                self.output_activation.gradient(self.z[i])
            else:
                delta = (delta @ self.w[i+1].T) * self.activation.gradient(self.z[i])
                
            # if first layer, calculate gradients of weighs from input data, 
            # otherwise, calculate using activations
            if i == 0:
                grad_w = (self.X_matrix.T @ delta) / self.n_datapoints
            else: 
                grad_w = (self.a[i-1].T @ delta) / self.n_datapoints
                
            # L2 Regularization
            grad_w += lmbda * self.w[i]

            # calculate gradient for bias
            grad_b = np.sum(delta, axis=0) / self.n_datapoints
            
            # update weights and biases
            self.w[i] -= (eta * grad_w)
            self.b[i] -= (eta * grad_b)
	
    
    def predict(self, X_test_matrix):
        '''
        Use _feed_forward() method to make predictions for X_test_matrix

        Returns:
            a(float):       Value of the output layer activation function  
        '''
        # Set input data to predict
        self.X_matrix = X_test_matrix
        self.n_datapoints = X_test_matrix.shape[0]
        # Predict
        self._feed_forward()

        return self.a[-1]


    def train(self, X_matrix, y_matrix, epochs, eta, lmbda, minibatch_n, info=True):
        '''
        Train the network using stochastic gradient descent with mini-batches
        of size minibatch_n. Thelearning rate is eta and the network is trained
        for n epochs. X_matrix is the training data, and y_matrix the target 
        data.
        '''

        if info: 
            print('Training network for {} epochs...'.format(epochs))
        
        # Number of mini_batches. Each has minibatch_n samples, except the 
        #last minibatch might have less samples
        n_samples = math.ceil( X_matrix.shape[0] / minibatch_n)
        
        train_loss = []
        for i in range(epochs):
            
            # shuffle and split into n_samples
            X, y = shuffle(X_matrix, y_matrix, random_state=42)
            X_resampled = np.array_split(X, n_samples)
            y_resampled = np.array_split(y, n_samples)
            
            iteration_losses = []
            
            for m in range(n_samples):
                
                # Set input data and labels to train with
                self.X_matrix = X_resampled[m]    
                self.y_matrix = y_resampled[m]
                self.n_datapoints = X_resampled[m].shape[0]
                # Calculate output, update weights and biases              
                self._feed_forward()
                self._backpropagate(eta, lmbda)
                # Calculate train loss
                loss = self.cost_function(self.y_matrix, self.a[-1])
                
                iteration_losses.append(loss.flatten())
                
            train_loss.append( np.mean(np.concatenate(iteration_losses)))
            
            if info: 
                print('Epoch {}, train loss {}'.format(i, loss))
                
            # End if difference from last train loss is less than 1e-5 and
            # we've trained for at least 10 epochs
            if i >= 10:
                do_break = True
                for n in range(10):
                    if not (train_loss[i-n-1] - train_loss[i-n]) < 1e-5:
                        do_break = False
                if do_break: 
                    break
            # End if train loss is nan
            if math.isnan(train_loss[-1]):
                self.ended_in_nan = True                
                break
            
