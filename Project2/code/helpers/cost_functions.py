# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:19:37 2021
@author: lidialuq

A choice of cost functions with their derivatives and preformance metrics
"""

import numpy as np


class MeanSquareError:
    def __str__(self):
        return 'Mean Square Error'
    
    def __call__(self, y, y_predicted):
        return np.mean(0.5*(y - y_predicted)**2)

    def gradient(self, y, y_predicted):
        return (y_predicted - y)

class BinaryCrossEntropy:
    def __str__(self):
        return 'Binary Cross Entropy'
    
    def __call__(self, y, y_predicted):
        return -np.mean( y * np.log(y_predicted) + (1-y) * np.log(1-y_predicted))

    def gradient(self, y, y_predicted):
        return (y_predicted -y)/(y_predicted* (1-y_predicted))
    
class R2:
    def __str__(self):
        return 'R2'
    
    def __call__(self, y, y_predicted):    
        sum_sq_error = np.sum((y - y_predicted) ** 2)
        sum_sq_res = np.sum((y - np.mean(y)) ** 2)
        return 1 - (sum_sq_error / sum_sq_res)

class Accuracy:
    def __str__(self):
        return 'Accuracy'
    
    def __call__(self, y, y_predicted):
        y_predicted = np.where(y_predicted >= 0.5, 1, 0)
        return np.mean(y == y_predicted)