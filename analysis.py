#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:33:14 2021
@author: lidia
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


class FrankeFunction():
    '''
    Class with all regression functionality (but statistics should probably be
    moved to another doc?)
    Args:
        x (1d array): x-coordinates 
        y (1d array): y-coordinates
        noise_var (float or int): variance of the noise
    '''
    def __init__(self, x = None, y = None, noise_var = 0, plot = False):
        # Generate x and y vectors divided equally in a 2d grid
        x = np.linspace(0, 1, 50) if not x else x
        y = np.linspace(0, 1, 50) if not y else y
        
        self.x_2d, self.y_2d = np.meshgrid(x,y)
        self.x = self.x_2d.ravel()
        self.y = self.y_2d.ravel()
        
        # Generate z, add noise
        z = self.franke_function(self.x, self.y)
        noise = np.random.normal(0,noise_var,(self.x.shape))
        self.z = z + noise
            
        # In order to plot the function, use x_2d and y_2d as input instead
        if plot: 
            z_2d = self.franke_function(self.x_2d, self.y_2d)
            noise = np.random.normal(0,noise_var,(self.x_2d.shape))
            z_2d = z_2d + noise
            self.plot(self.x_2d, self.y_2d, z_2d)
            
    def franke_function(self, x, y):
        '''
        Returns the values of the franke function evaluated at x,y
        The shape of the returned z array is equal to the shape of the x and y 
        inputs.
        '''
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4
        
    def plot(self, x_2d, y_2d, z_2d):

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
  


class LinearRegression():
    '''
    Methods needed for linear regression, currently only OLS is implemented
    Args:
        x (1d array): x-coordinates 
        y (1d array): y-coordinates
        z_data (1d array): z vector created by FrankeFunction()
        degree (int): degree of polynomial to use in regression
        split (bool): split the data in train and test
        model (str): model to use in regression (only 'OLS' implemented)
        seed: seed to split the data in order to get reproducable results
    '''
    def __init__(self, x, y, z_data, degree = 3, split = False, test_size = 0.2,
                 model = 'OLS', seed = 42):
        
        # NOTE: This is pretty disorganised, sorry... Initiating an instance  
        # will automatically calculate beta and z_model for model (only OLS 
        # implemented) with or without split BUT in order to use bootstrap or 
        # kfold the method has to be called explicitely (see end of code)
        
        self.degree = degree
        self.x = x
        self.y = y
        self.z_data = z_data
        self.X = self.design_matrix()
        
        if split: 
            self.X_train, self.X_test, self.z_train, self.z_test = \
            train_test_split(self.X, self.z_data, test_size, 
                             shuffle=False, random_state = seed)  
            if model == 'OLS':
                self.beta = self.OLS(self.X_train, self.z_train)
                self.z_model = self.predict(self.X_test, self.beta)
        else:
            if model == 'OLS':
                self.beta = self.OLS(self.X, self.z_data)
                self.z_model = self.predict(self.X, self.beta)
            
                
    def OLS(self, X, z):
        '''
        Get parameters for ordinary least squares regression
        Args: 
            X (2d array): design matrix
            z (1d array): z vector (input to model)
        Return: 
            beta (1d array): regression parameters
        '''
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        return beta
    
        
    def design_matrix(self):
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
        for y_exp in range(self.degree +1):
            for x_exp in range(self.degree +1):
                if y_exp + x_exp <= self.degree:
                    X.append(self.x**x_exp * self.y**y_exp)
        X = (np.array(X).T).squeeze()
        return X

    def predict(self, X, beta):
        '''
        Predicts z values from regression parameters and the design matrix
        Args: 
            X (2d array): design matrix
            beta (1d array): regression parameters
        Return: 
            z (1d array): predicted z-values
        '''
        return (X @ beta)
        
    def plot(self, x_2d, y_2d, z):
        '''
        Plot a z vector (after reshaping) given a x and y mesh.
        Can be used to plot z_model in order to visualize results
        NOTE: only works when split = False (otherwise x_2d and y_2d do not
        match size of z after reshaping)
        '''
        # Reshape z vector into 2d matrix to plot
        try:
            z_2d = z.reshape(x_2d.shape[1], x_2d.shape[0])
        except ValueError: 
            print('Model could not be plotted. Split must be set to False \
                  in order to plot model')
            return

        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x_2d, y_2d, z_2d, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_title('Model')
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    
        
    def MSE(self, z_test, z_model):
        sun_sq_error = np.sum((z_test - z_model) ** 2)
        
        return sun_sq_error / np.size(z_model)
    
    def R2(self, z_test, z_model):
        sum_sq_error = np.sum((z_test - z_model) ** 2)
        sum_sq_res = np.sum((z_test - np.mean(z_test)) ** 2)
        
        return 1 - (sum_sq_error / sum_sq_res)
    
    def bias(self, z_test, z_model):
        
        return np.mean((z_test - z_model)**2)
    
    def var(self, z_model):
        return np.mean(np.var(z_model))
        
    def beta_variance(self, X, sigma_sq = 1):
        return sigma_sq * (np.linalg.pinv(X.T @ X))
    
    def bootstrap(self, X_train, X_test, z_train, z_test, bootstrap_nr=None, 
                  model='OLS', lamb=0):
        """
        Bootstrap sampling. 
        Args: 
            X_train (2d array): design matrix
            z_train (1d array): train z vector
            z_test (1d array): test z vector
            bootstrap_nr: number of bootstraps, defaults to len(z_train)
            model (str): Regression model used. 'OLS', 'Ridge', or 'Lasso' 
                (only OLS currently implemented). Defaults ot OLS. 
            lamb (float): value of lambda for Ridge and Lasso regressions
            
        Returns: 
            z_train (list[1d array]): List of arrays of len(z_train). The list 
                has length = bootstrap_nr
            z_model
            z_fit
        """
        
        if not bootstrap_nr: bootstrap_nr = len(z_train)
        
        z_train_boot = []
        z_model_boot = []
        z_fit_boot = []
        for i in range(bootstrap_nr):
            
            tmp_X_train, tmp_z_train = resample(X_train, z_train, replace=True)
            if model == "OLS": tmp_beta = self.OLS(tmp_X_train, tmp_z_train)
            if model == "Ridge": pass
            if model == "Lasso": pass
            
            z_train_boot.append(tmp_z_train)
            z_model_boot.append(X_test @ tmp_beta)
            z_fit_boot.append(tmp_X_train @ tmp_beta)
 
    
        return z_train_boot, z_model_boot, z_fit_boot
    
    
    
    def kfold(self, X, z, k=5, model="OLS", lamb=0):
        """
        Cross-validation 
        Args: 
            X (2d array): design matrix
            z (1d array): z-vector
            k (int): number of folds to divide data into
            model (str): Regression model used. 'OLS', 'Ridge', or 'Lasso' 
                (only OLS currently implemented). Defaults ot OLS. 
            lambda (float): value of lambda for Ridge and Lasso regressions
            
        Returns: 
            z_train (list[1d array]): List of arrays of len(z_train). The list 
                has length = bootstrap_nr
            z_model ((list[1d array]): List of len=bootstrap_nr of arrays of 
                     len(z_train)
            z_model ((list[1d array]): List of len=bootstrap_nr of arrays of 
                     len(z_test)
        """
        
        # shuffle X and z together (so that they still match)
        X, z = shuffle(X, z, random_state=42)
        # split into k vectors
        X_folds = np.array_split(X, k)
        z_folds = np.array_split(z, k)
        
        z_train = []
        z_model = []
        z_fit = []
        for i in range(k):
            # test vectors are fold index i, train vectors the rest
            tmp_X_test = X_folds[i]
            tmp_z_test = z_folds[i]
            tmp_X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
            tmp_z_train = np.concatenate(z_folds[:i] + z_folds[i+1:])
            
            # regression
            if model == "OLS": tmp_beta = self.OLS(tmp_X_train, tmp_z_train)
            if model == "Ridge": pass
            if model == "Lasso": pass
        
            z_train.append(tmp_z_train)
            z_model.append((tmp_X_test @ tmp_beta).ravel())
            z_fit.append((tmp_X_train @ tmp_beta).ravel())
        
        return z_train, z_model, z_fit
            

if __name__ == '__main__':
    
    # Get input data, plot function 
    ff = FrankeFunction(noise_var = 0.05, plot=True)
    # Linear regression
    ols = LinearRegression(ff.x, ff.y, ff.z, degree = 5, split=False)
    ols.plot(ff.x_2d, ff.y_2d, ols.z_model)
    
#    # Examples of using bootstrap and kfold sampling:
#    # NOTE: split arg in LinearRegression() must be set to True in order to
#    # use bootstrap() and kfold()
#    z_train, z_model, z_fit = ols.bootstrap(ols.X_train, ols.X_test, 
#                                            ols.z_train, ols.z_test, 
#                                            bootstrap_nr=3)
#    z_train, z_model, z_fit = ols.kfold(ols.X, ols.z_data, k=3)
