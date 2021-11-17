import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import time

from sgd import SGD
from data.franke_function import FrankeFunction, get_ff_data
from data.breast_cancer import get_breastcancer
from helpers.cost_functions import MeanSquareError, Accuracy
from helpers.activations import Sigmoid, Linear

from analysis_proj1 import LinearRegression
from sklearn.linear_model import LogisticRegression


def main():
    np.random.seed(42)

    grad_meth = 'ridge'
    act_func = Sigmoid()
    score = Accuracy()

    start = time.process_time()


    #N = int(20)
    #deg = 3
    #ff = FrankeFunction(axis_n = N, noise_var = 0.1, plot = False)

    #ols = LinearRegression(ff.x, ff.y, ff.z, degree = deg, split=True, test_size = 0.3, plot=False)
    #print('ols',score(ols.z_test, ols.z_model))

    #print('time', time.process_time() - start, 's')

    #X_train, X_test, z_train, z_test = get_ff_data(axis_n = N, degree = deg) # calls ff(axis_n = 20, noise_var = 0.1, plot = False)
    X_train, X_test, z_train, z_test = get_breastcancer()
    z_train = np.expand_dims(z_train, axis = 1)
    z_test = np.expand_dims(z_test, axis = 1)

    etas = np.logspace(-4,1,10)
    lmds = np.logspace(-6,1,5)
    mbs = np.linspace(0,20,5)
    gammas = np.linspace(0.001,0.9999,6)
    beta1s = np.linspace(0.9, 0.999, 5)
    beta2s = np.linspace(0.99,0.9999, 5)

    sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                eta=etas[6], lmd = lmds[0], epochs=100, minibatch_n = int(mbs[4]))
       
    sgd(sgd_method='vanilla')
    zvan_pred = (sgd.predict())
    print(f'sgd_method {score}')
    print('vanilla',score(z_test, act_func(X_test @ sgd.theta)))
    print('vanilla',score(z_train, zvan_pred))
    print('time', time.process_time() - start, 's')
    
    sgd.gamma = 0.4
    sgd(sgd_method='momentum')
    zmom_pred = sgd.predict()
    print('momentum',score(z_test, act_func(X_test @ sgd.theta)))
    print('momentum',score(z_train, zmom_pred))
    print('time', time.process_time() - start, 's')

    sgd.beta1 = beta1s[1]
    sgd.beta2 = beta2s[2]
    sgd(sgd_method='Adam')
    zAdam_pred = sgd.predict()
    print('Adam',score(z_test, act_func(X_test @ sgd.theta)))
    print('Adam',score(z_train, zAdam_pred))
    #print(np.sort(zAdam_pred.T))
    #print(z_train.T == np.around(zAdam_pred.T))
    print('time', time.process_time() - start, 's')
        
if __name__ == '__main__':
    main()