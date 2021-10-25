# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:19:37 2021
@author: lidialuq

A flexible Feed Forward Neural Network class that can be used for both
regression and classification problems, with a choice of activation functions
and cost functions.
"""
 
   
import numpy as np

class FFNN:

    def __init__(self, 
             n_datapoints: int,
             n_input_neurons: int,
             n_output_neurons: int, 
             hidden_layers_struct: list, 
             activation,   
             output_activation,
             cost_function,
             ):
        
        # Values that define structure of neural network
        self.n_datapoints = n_datapoints
        self.n_output_neurons = n_output_neurons
        self.n_input_neurons = n_input_neurons
        self.hidden_layers_struct = hidden_layers_struct
        self.n_hidden_layers = len(hidden_layers_struct)
        self.activation = activation
        self.output_activation = output_activation
        self.cost_function = cost_function
        
        # Create starting weights and biases
        self.w = self._initialize_weights()
        self.b = self._initialize_biases()
        self.a = self._initialize_activations()
        self.z = self._initialize_activations()
		


    def _initialize_weights(self):
        '''
        Initialize weights with a normal distribution (mean 0, var 1)
        Returns a list containing 2d nd.nparrays with the weights for each layer
        '''
        w = []
        # Weights of input neurons
        w.append( np.random.randn(self.n_input_neurons, 
                                  self.hidden_layers_struct[0]))
        
        # Weights of neurons between hidden layers (if only 1 hidden layer, 
        # nothing is appended)
        for i in range(0, self.n_hidden_layers-1):
            w.append( np.random.randn(self.hidden_layers_struct[i], 
                                         self.hidden_layers_struct[i+1]))
        # Weights of last hidden layer
        w.append( np.random.randn(self.hidden_layers_struct[-1], 
                                  self.n_output_neurons))
        
        return w
    
    def _initialize_biases(self):
        '''
        Initialize biases to zero
        Returns a list containing 1d np.ndarrays with the biases for each layer 
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
        equal to the input data, all other layers initialized to None
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
        """
    
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


    def _backpropagate(self, eta):
        '''
        Update weights and gradients for each layer by calculating the 
        gradients of the cost function (here assumed to be mean square error) 
        The gradient of the cost function w.r.t z (here called error) is used
        to calculate the gradients w.r.t the weights and the biases.
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

            # calculate gradient for bias
            grad_b = np.sum(delta, axis=0) / self.n_datapoints
            
            # update weights and biases
            self.w[i] = self.w[i] - (eta * grad_w)
            self.b[i] = self.b[i] - (eta * grad_b)
	
    
    def predict(self, X_test_matrix):
        '''
        Use _feed_forward() method to make predictions for X_test_matrix
        '''
        
        self.X_matrix = X_test_matrix
        self._feed_forward()

        return self.a[-1]


    def train(self, X_matrix, y_matrix, epochs, eta, info=True):
        '''
        Train the network with learning rate eta for n epochs. 
        X_matrix is the training data, and y_matrix the target data.
        '''
        self.X_matrix = X_matrix    # (n_datapoints x n_features)
        self.y_matrix = y_matrix    # (n_datapoints x n_output_neurons)
        
        if info: 
            print('Training network for {} epochs...'.format(epochs))
        train_loss = []
        for i in range(epochs):  
            # Calculate output, update weights and biases              
            self._feed_forward()
            self._backpropagate(eta)
            # Calculate train loss
            loss = self.cost_function(self.y_matrix, self.a[-1])
            train_loss.append(loss)
            
            if info: 
                print('Epoch {}, train loss {}'.format(i, loss))