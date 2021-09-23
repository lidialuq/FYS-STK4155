#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:33:14 2021

@author: lidia
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split


class FrankeFunction():
    '''
    Creates and plots the input data for exercices 1-5. 
    Args:
        x (1d array): x-coordinates 
        y (1d array): y-coordinates
        noise_factor (float or int): factor the noise is muliplied by 
    '''
    def __init__(self, x = None, y = None, noise_factor = 0):
        self.x = np.linspace(0, 1, 2000) if not x else x
        self.y = np.linspace(0, 1, 2000) if not y else y
        self.x_2d, self.y_2d = np.meshgrid(self.x,self.y)
        self.noise_factor = noise_factor
        
        term1 = 0.75*np.exp(-(0.25*(9*self.x_2d-2)**2) - 0.25*((9*self.y_2d-2)**2))
        term2 = 0.75*np.exp(-((9*self.x_2d+1)**2)/49.0 - 0.1*(9*self.y_2d+1))
        term3 = 0.5*np.exp(-(9*self.x_2d-7)**2/4.0 - 0.25*((9*self.y_2d-3)**2))
        term4 = -0.2*np.exp(-(9*self.x_2d-4)**2 - (9*self.y_2d-7)**2)
        
        noise = np.random.normal(0,1,(self.x_2d.shape))
        self.z = term1 + term2 + term3 + term4 \
            + (noise * self.noise_factor)
        
    def plot(self):
        '''
        Works for now but have to make it nicer for report
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.x_2d, self.y_2d, self.z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
  


class LinearRegression():
    '''
    Methods needed for linear regression, currently only OLS is implemented
    Args:
        x (1d array): x-coordinates 
        y (1d array): y-coordinates
        z_data (2d array): z matrix created by FrankeFunction()
        degree (int): degree of polynomial to use in regression
        split (bool): split the data in train and test
        model (str): model to use in regression (only 'OLS' implemented)
        seed: seed to split the data in order to get reproducable results
        
        NOTE: somthing here is clearly not working as seen from the plot of 
        z_model...
    '''
    def __init__(self, x, y, z_data, degree = 2, split = False, model = 'OLS', 
                 seed = 42):
        self.degree = degree
        self.x = x
        self.y = y
        self.x_2d, self.y_2d = np.meshgrid(self.x ,self.y)
        self.z_data = z_data
        self.X = self.design_matrix()
        
        if split: 
            self.X_train, self.X_test, self.z_train, self.z_test = \
            train_test_split(self.X, self.z_data, test_size=0.2, 
                             shuffle=False, random_state = seed)  
            if model == 'OLS':
                self.beta = self.OLS(self.X_train, self.z_train)
                self.z_model = self.predict(self.X_test, self.beta)
        else:
            if model == 'OLS':
                self.beta = self.OLS(self.X, self.z_data)
                self.z_model = self.predict(self.X, self.beta)
            
        
    def OLS(self, X, z):
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
        return (X @ beta)
        
    def plot(self, z):
        # Create new x, y axis with right length
        x = np.linspace(0,1,ols.z_model.shape[1])
        y = np.linspace(0,1,ols.z_model.shape[0])
        x_2d, y_2d = np.meshgrid(x,y)
        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x_2d, y_2d, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    def MSE(self):
        sun_sq_error = np.sum((self.z_test - self.z_model) ** 2)
        
        return sun_sq_error / np.size(self.z_model)
    
    def R2(self):
        sum_sq_error = np.sum((self.z_test - self.z_model) ** 2)
        sum_sq_res = np.sum((self.z_test - np.mean(self.z_test)) ** 2)
        
        return 1 - (sum_sq_error / sum_sq_res)
    
    def variance(self, X, sigma_sq = 1):
        var1 = sigma_sq * (np.linalg.pinv(X.T @ X))
        # z_1D = np.ravel(z_data)
        # z_model_1D = np.ravel(z_model)
        # n = len(z_1D)
        # sigma_sq = (1/(n-self.degree-1)) * sum((z_1D - z_model_1D)**2)
        
        return var1, sigma_sq
    

if __name__ == '__main__':
        
    ff = FrankeFunction()
    ff.plot()
    ols = LinearRegression(ff.x, ff.y, ff.z, 3, split=True)
    ols.plot(ols.z_model)

