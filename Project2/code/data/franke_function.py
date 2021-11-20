# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:00:38 2021
@author: lidialuq
"""
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split

np.random.seed(seed=42) 

class FrankeFunction():
    '''
    Class to sample franke function

    Attributes:
        axis_n (int):               number of datapoint per axis (total n = axis_n^2)
        noise_var (float or int):   variance of the noise
        plot (bool):                True if the function is to be plotted. Default False.

    Methods:
        design_matrix():
            "Creates the design matrix"
    '''
    def __init__(self, axis_n = 20, noise_var = 0, plot = False):
        # Generate x and y vectors divided equally in a 2d grid
        x = np.linspace(0, 1, axis_n) 
        y = np.linspace(0, 1, axis_n)
        
        x_2d, y_2d = np.meshgrid(x,y)
        self.x = x_2d.ravel()
        self.y = y_2d.ravel()
        
        # Generate z, add noise
        z = self._franke_function(self.x, self.y)
        noise = np.random.normal(0,noise_var,(self.x.shape))
        self.z = z + noise
            
        # In order to plot the function, use x_2d and y_2d as input instead
        if plot: 
            z_2d = self._franke_function(x_2d, y_2d)
            noise = np.random.normal(0,noise_var,(x_2d.shape))
            z_2d = z_2d + noise
            self.plot(x_2d, y_2d, z_2d)
            
    def _franke_function(self, x, y):
        '''
        Returns the values of the franke function evaluated at x,y
        The shape of the returned z array is equal to the shape of the x and y 
        inputs.

        Returns:
            franke (2d array): output of the Franke Function
        '''
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4
        
    def _plot(self, x_2d, y_2d, z_2d):
        '''
        Plots the Franke Function.
        '''

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x_2d, y_2d, z_2d, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_title('Original data')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    def design_matrix(self, x, y, degree):
        '''
        Creates the design matrix

        Args:
            x (1d array): x-coordinates 
            y (1d array): y-coordinates
            degree (int): degree of polynomial in model

        Returns: 
            X (2d array): design matrix
        '''

        X = []
        for y_exp in range(degree +1):
            for x_exp in range(degree +1):
                if y_exp + x_exp <= degree:
                    X.append(x**x_exp * y**y_exp)
        X = (np.array(X).T).squeeze()
        return X
    
def get_ff_data(axis_n = 20, degree=3):
    '''
    Initializes an instance of the FrankeFunction class, creating the data,
    and splitting it into test and train sets along with the respective 
    design matrices X.

    Args:
        axis_n (int): Optional. number of datapoint per axis (total n = axis_n^2).
            Default 20.
        degree (int): Optional. polynomial degree of the target design matrix. 
            Default 3.

    Returns:
        X_train (2d array): train part of the design matrix X
        X_test (2d array): test part of the design matrix X
        z_train (2d array): train part of the Franke data z
        z_test (2d array): train part of the Franke data z
    '''

    np.random.seed(42)
    
    # create data
    ff = FrankeFunction(axis_n = axis_n, noise_var = 0.1, plot = False)
    X = ff.design_matrix(ff.x, ff.y, degree = degree)
    
    # split data 
    X_train, X_test, z_train, z_test = \
        train_test_split(X, ff.z, test_size = 0.3, shuffle = True)
    
    # normalize by removing mean
    for i in range(X_train.shape[1]):
        X_train[:,i] = X_train[:,i] - np.mean(X_train[:,i])
        X_test[:,i]  = X_test[:,i] - np.mean(X_test[:,i])
    z_train = z_train - np.mean(z_train)
    z_test = z_test - np.mean(z_test)
    
    # change shape from (datapoints,) to (datapoints,1)
    z_train = np.expand_dims(z_train, 1)
    z_test = np.expand_dims(z_test, 1)
    return X_train, X_test, z_train, z_test

