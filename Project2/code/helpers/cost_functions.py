# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:19:37 2021
@author: lidialuq

A choice of cost functions with their derivatives
"""

import numpy as np


class MeanSquareError:
    def __str__(self):
        return 'Mean Square Error'
    
    def __call__(self, y, y_predicted):
        return np.mean(0.5*(y - y_predicted)**2)

    def gradient(self, y, y_predicted):
        return y - y_predicted

class BinaryCrossEntropy:
    def __str__(self):
        return 'Binary Cross Entropy'
    
    def __call__(self, y, y_predicted):
        return -y*np.log(y_predicted) + (1-y)*np.log(1-y_predicted) 

    def gradient(self, y, y_predicted):
        return (y_predicted - y)/(y_predicted * (1-y_predicted))