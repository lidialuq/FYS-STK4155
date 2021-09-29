# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:58:52 2021

@author: lidia
"""
from analysis import FrankeFunction, LinearRegression
import matplotlib.pyplot as plt
import numpy as np


def plot_conf_interval():
    '''
    Exercice 1
    Plot betas with confidence intervals. Get MSE and R2
    '''
    noise_var = 0.05
    axis_n = 10
    degree= 5
    ff = FrankeFunction(axis_n = axis_n, noise_var = noise_var, plot=False)
    ols = LinearRegression(ff.x, ff.y, ff.z, degree = degree, split=False)
    beta_var = ols.beta_variance(ols.X, sigma_sq = noise_var)
    x = list(range(0, len(beta_var)))
    beta_conf = 1.96*np.sqrt(beta_var/axis_n**2)
    
    plt.errorbar(x, ols.beta, yerr=beta_conf, markersize=4, linewidth=1, \
                 ecolor="black", fmt='o', capsize=5, capthick=1)
    plt.title("Regression parameters")
    x_labels = [r"$\beta_"+"{{{:}}}$".format(i) for i in range(len(ols.beta))]
    plt.xticks(x, x_labels)
    plt.show()
    
    MSE = ols.MSE(ols.z_data, ols.z_model)
    R2 = ols.R2(ols.z_data, ols.z_model)
    print("MSE = {}".format(MSE))
    print("R2 = {}".format(R2))
    

def plot_train_test_MSE():
    '''
    Exercice 2
    Plot test and train MSE. FOr "nice" plot, axis_n=10, noise_var=0.05, 
    # deg range(1,9)
    '''
    
    ff = FrankeFunction(axis_n = 10, noise_var = 0.05, plot=False)
    trainMSE = []
    testMSE = []
    x = []
    for deg in range(1,9):
        x.append(deg)
        ols = LinearRegression(ff.x, ff.y, ff.z, degree = deg, split=True, 
                               test_size =0.3)
        trainMSE.append( ols.MSE(ols.z_train, ols.z_model_train))
        testMSE.append( ols.MSE(ols.z_test, ols.z_model))
        
    plt.plot(x, testMSE, 'b') 
    plt.plot(x, trainMSE, 'r') 
    plt.legend(["Test MSE", "Train MSE"])
    plt.title("OLS MSE")
    plt.xlabel("Model complexity (degree)")
    plt.ylabel("MSE")
    plt.show()
    
def plot_bias_var_bootstrap():
    '''
    Plot the mean of the bias, variance and MSE over samples of the data for 
    The sampling is done using the bootstrap method. 
    '''
    axis_n = 20 # nr of datapoints in one axis. Total n is axis_n**2
    bootstrap_nr = 100  #nr of samples
    max_degree = 10
    ff = FrankeFunction(axis_n = axis_n, noise_var = 0.05, plot=False)
    MSE, bias, var = [],[],[]
    x = []
    
    for deg in range(1,max_degree):
        x.append(deg)
        ols = LinearRegression(ff.x, ff.y, ff.z, degree = deg, split=True, 
                               test_size =0.3)
        MSE_, bias_, var_ = ols.bootstrap(ols.X_train, ols.X_test, 
                                          ols.z_train, ols.z_test,
                                          bootstrap_nr=bootstrap_nr)
        MSE.append(MSE_)
        bias.append(bias_)
        var.append(var_)      
        
    plt.plot(x, MSE, 'b') 
    plt.plot(x, bias, 'r') 
    plt.plot(x, var, 'g') 
    plt.yscale('log')
    plt.legend(["MSE", "bias", "var"])
    plt.title("Bias-variance (bootstrap)")
    plt.xlabel("Model complexity (degree)")
    plt.show()
    
if __name__ == '__main__':
    
    #plot_train_test_MSE()
    #plot_conf_interval()
    plot_bias_var_bootstrap()
