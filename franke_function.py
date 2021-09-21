#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:33:14 2021

@author: lidia
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class FrankeFunction():
    def __init__(self, x = None, y = None, noise_factor = 0):
        self.x = np.sort(np.random.uniform(0,1,200)) if not x else x
        self.y = np.sort(np.random.uniform(0,1,200)) if not x else x
        self.x_2d, self.y_2d = np.meshgrid(self.x,self.y)
        self.noise_factor = noise_factor
        
        term1 = 0.75*np.exp(-(0.25*(9*self.x_2d-2)**2) - 0.25*((9*self.y_2d-2)**2))
        term2 = 0.75*np.exp(-((9*self.x_2d+1)**2)/49.0 - 0.1*(9*self.y_2d+1))
        term3 = 0.5*np.exp(-(9*self.x_2d-7)**2/4.0 - 0.25*((9*self.y_2d-3)**2))
        term4 = -0.2*np.exp(-(9*self.x_2d-4)**2 - (9*self.y_2d-7)**2)
        
        noise = np.random.normal(0,1,(self.x_2d.shape))
        self.z = term1 + term2 + term3 + term4 \
            + (self.noise_factor * noise)
        
    def plot(self):
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
  


class Train():
    def __init__(self, x, y, degree):
        self.degree = degree
        self.x = x
        self.y = y
        
    def OLS(self, X, z):
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        return beta
    
    def OLS_svd(self, X, z):
        pass
        
    def design_matrix(self, x, y, degree):
        '''
        Creates the design matrix
        Args:
            x (1d array): x-coordinates 
            x (1d array): x-coordinates
            degree (int): degree of polynomial in model
        Returns: 
            X (2d array): design matrix
            xy_exp (list): a list of the exponentials in the order used in the 
            design matrix. This will be used to evaluate the resulting polynomial.
        '''
        X = []
        for y_exp in range(self.degree +1):
            for x_exp in range(self.degree +1):
                if y_exp + x_exp <= self.degree:
                    X.append(self.x**x_exp * self.y**y_exp)
        X = (np.array(X).T).squeeze()
        return X
    
    # def evaluate_poly(self, x, y, beta): 
    #     z = np.zeros(x.shape)
    #     beta_idx = 0
    #     for y_exp in range(self.degree +1):
    #         for x_exp in range(self.degree +1):
    #             if y_exp + x_exp <= self.degree:
    #                 z += beta[beta_idx] * x**x_exp * y**y_exp
    #                 beta_idx += 1
    #     return z

    def predict(self, beta, X):
        return (X @ beta)
        
    def plot(self, x, y, z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


class Evaluate():
    def __init__(self, y, y_predicted, X, beta):
        self.y = y
        self.y_predicted = y_predicted
        self.X = X
        self.beta = beta
    
    def MSE(self):
        sun_sq_error = np.sum((self.y - self.y_predicted) ** 2)
        
        return sun_sq_error / np.size(self.y_predicted)
    
    def R2(self):
        sum_sq_error = np.sum((self.y - self.y_predicted) ** 2)
        sum_sq_res = np.sum((self.y - np.mean(self.y)) ** 2)
        
        return 1 - (sum_sq_error / sum_sq_res)
    
    def var(self):
        var = np.var(self.y) * np.linalg.pinv(self.X.T @ self.X)
        return var
        
ff = FrankeFunction()
ff.plot()
train = Train(ff.x, ff.y, 2)
X = train.design_matrix(ff.x, ff.y, 2)    
beta = train.OLS(X, ff.z)
z = train.predict(beta, X)
train.plot(ff.x_2d, ff.y_2d, z)
evaluate = Evaluate(ff.z, z, X, beta)
MSE = evaluate.MSE()
R2 = evaluate.R2()
var = evaluate.var()
