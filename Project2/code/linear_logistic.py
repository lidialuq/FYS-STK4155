import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import time

from sgd import SGD
from data.franke_function import FrankeFunction, get_ff_data
from data.breast_cancer import get_breastcancer
from helpers.cost_functions import MeanSquareError, Accuracy, R2
from helpers.activations import Sigmoid, Linear

from analysis_proj1 import LinearRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier


def linear(grad_meth, score = R2()):
    #np.random.seed(42)

    grad_meth = 'ordinary' # 'ordinary' or 'ridge'
    act_func = Linear()
    score = R2() # 'R2()' or 'MeanSquareError()'
    print(f'Using grad_meth = {grad_meth}, act_func = {act_func} and score = {score}')

    if grad_meth == 'ordinary':
        mod = 'OLS'
    elif grad_meth == 'ridge':
        mod = 'Ridge'

    N = int(20)
    deg = 3
    ff = FrankeFunction(axis_n = N, noise_var = 0.1, plot = False)

    lin = LinearRegression(ff.x, ff.y, ff.z, degree = deg, split=True, test_size = 0.3, plot=False, model=mod, lamb=1e-4)
    lin_test = []
    lin_test.append(score(lin.z_test, lin.z_model))

    print('time', time.process_time() - start, 's')

    X_train, X_test, z_train, z_test = get_ff_data(axis_n = N, degree = deg) # calls ff(axis_n = 20, noise_var = 0.1, plot = False)
    
    etas = np.logspace(-4,1,10)
    lmds = np.logspace(-6,1,5)
    mbs = np.linspace(1,20,5)
    gammas = np.linspace(0.001,0.9999,6)
    beta1s = np.linspace(0.9, 0.999, 5)
    beta2s = np.linspace(0.99,0.9999, 5)

    sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                eta=etas[6], lmd = lmds[1], epochs=100, minibatch_n = int(mbs[2]))
    

    van_test, van_train, mom_test, mom_train, Adam_test, Adam_train \
    = [], [], [], [], [], []

    # initialize best vanilla values
    sgd.eta = etas[8]
    sgd.lmd = lmds[1]
    sgd.minibatch_n = int(mbs[4])
    sgd(sgd_method='vanilla') # fit model
    van_test.append(score(z_test, sgd.predict(X_test)))
    van_train.append(score(z_train, sgd.predict(X_train)))
    
    # initialize best momentum values
    sgd.eta = etas[8]
    sgd.lmd = lmds[1]
    sgd.minibatch_n = int(mbs[4])
    sgd.gamma = gammas[4] 
    sgd(sgd_method='momentum') # fit model
    mom_test.append(score(z_test, sgd.predict(X_test)))
    mom_train.append(score(z_train, sgd.predict(X_train)))

    # initialize best Adam values
    sgd.eta = etas[6]
    sgd.lmd = lmds[1]
    sgd.minibatch_n = int(mbs[4])
    sgd.beta1 = beta1s[2]
    sgd.beta2 = beta2s[4]
    sgd(sgd_method='Adam') # fit model
    Adam_test.append(score(z_test, sgd.predict(X_test)))
    Adam_train.append(score(z_train, sgd.predict(X_train)))

    return np.array((Ridge_test,van_test, van_train, mom_test, mom_train, Adam_test, Adam_train))

        
def logistic(grad_meth):
    #np.random.seed(42)

    grad_meth = 'ordinary'
    act_func = Sigmoid()
    score = Accuracy()
    print(f'Using grad_meth = {grad_meth}, act_func = {act_func} and score = {score}')

    X_train, X_test, z_train, z_test = get_breastcancer()
    z_train = np.expand_dims(z_train, axis = 1)
    z_test = np.expand_dims(z_test, axis = 1)

    etas = np.logspace(-4,1,10)
    lmds = np.logspace(-6,1,5)
    mbs = np.linspace(1,20,5)
    gammas = np.linspace(0.001,0.9999,6)
    beta1s = np.linspace(0.9, 0.999, 5)
    beta2s = np.linspace(0.99,0.9999, 5)

    sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                eta=etas[9], lmd = lmds[2], epochs=1000, minibatch_n = int(mbs[3]))
    
    van_test, van_train, mom_test, mom_train, Adam_test, Adam_train \
    = [], [], [], [], [], []

    # initialize best vanilla values
    sgd.eta = etas[5] 
    sgd.lmd = lmds[0] 
    sgd.minibatch_n = int(mbs[3])
    sgd(sgd_method='vanilla')
    van_test.append(score(z_test, sgd.predict(X_test)))
    van_train.append(score(z_train, sgd.predict(X_train)))
    
    # initialize best momentum values
    sgd.eta = etas[4] 
    sgd.lmd = lmds[0] 
    sgd.gamma = gammas[4] 
    sgd.minibatch_n = int(mbs[1]) 
    sgd(sgd_method='momentum')
    mom_test.append(score(z_test, sgd.predict(X_test)))
    mom_train.append(score(z_train, sgd.predict(X_train)))


    # initialize best Adam values
    sgd.eta = etas[4] 
    sgd.lmd = lmds[2] 
    sgd.minibatch_n = int(mbs[0]) 
    sgd.beta1 = beta1s[1]
    sgd.beta2 = beta2s[2]
    sgd(sgd_method='Adam')
    Adam_test.append(score(z_test, sgd.predict(X_test)))
    Adam_train.append(score(z_train, sgd.predict(X_train)))
    
  
    return np.array((van_test, van_train, mom_test, mom_train, Adam_test, Adam_train))

def skl_log():

    X_train, X_test, z_train, z_test = get_breastcancer()

    clf = LogisticRegression(penalty='none').fit(X_train,z_train)
    #clf = SGDClassifier(loss='log',penalty='none').fit(X_train,z_train)
    pred = clf.predict(X_test)
    print(Accuracy()(pred,z_test), 'sklearn', 'penalty',clf.penalty)

if __name__ == '__main__':
    print('Disable ALL random.seed, also from imported files and functions!')
    a = []
    b = []
    k = 10 # number of runs
    
    # ugly stuff, but works
    for i in range(k):
        a.append(linear())
        b.append(logistic())
    def lin():



    a = np.array(a).reshape(k,7)
    average = np.mean(a,axis=0) # multiply by 100 for percent acc
    Ridge_test = average[0]
    van_test = average[1]
    van_train = average[2]
    mom_test = average[3]
    mom_train = average[4]
    Adam_test = average[5]
    Adam_train = average[6]
    print(lin_test,van_test, van_train, mom_test, mom_train, Adam_test, Adam_train)
   

    b = np.array(b).reshape(k,6)
    average = np.mean(b,axis=0)
    van_test = average[0]
    van_train = average[1]
    mom_test = average[2]
    mom_train = average[3]
    Adam_test = average[4]
    Adam_train = average[5]
    print(van_test, van_train, mom_test, mom_train, Adam_test, Adam_train)
