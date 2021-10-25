# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:43:02 2021
@author: lidialuq

A choice of activation functions with their derivatives
"""

import numpy as np


class Linear:
    def __str__(self):
        return 'Linear'
    
    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1

class Sigmoid:
    def __str__(self):
        return 'Sigmoid'
    
    def __call__(self, x):
        return 1/(1 + np.exp(-x))

    def gradient(self, x):
        return np.exp(-x)/(1 + np.exp(-x))**2
    
class ReLu:
    def __str__(self):
        return 'ReLu'
    
    def __call__(self, x):
        return np.maximum(x, 0) 
    
    def gradient(self, x):
        return np.where( x<0, 0, 1)
    
class LeakyReLu:
    def __str__(self):
        return 'LeakyReLu'
    
    def __call__(self, x):
        return np.where( x<0, 0.01*x, x)
    
    def gradient(self, x):
        return np.where( x<0, 0.01, 1)
    
class Tanh:
    def __str__(self):
        return 'Tanh'
    
    def __call__(self, x):
        return np.tanh(x)
    
    def gradient(self, x):
        return 1 - np.tanh(x)**2