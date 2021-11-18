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


def linear():
    np.random.seed(42)

    grad_meth = 'ordinary'
    act_func = Linear()
    score = MeanSquareError()

    start = time.process_time()


    N = int(50)
    deg = 3
    ff = FrankeFunction(axis_n = N, noise_var = 0.1, plot = False)

    Ridge = LinearRegression(ff.x, ff.y, ff.z, degree = deg, split=True, test_size = 0.3, plot=False, model='Ridge')
    print('ridge',score(Ridge.z_test, Ridge.z_model))

    print('time', time.process_time() - start, 's')

    X_train, X_test, z_train, z_test = get_ff_data(axis_n = N, degree = deg) # calls ff(axis_n = 20, noise_var = 0.1, plot = False)
    
    etas = np.logspace(-4,1,10)
    lmds = np.logspace(-6,1,5)
    mbs = np.linspace(1,20,5)
    gammas = np.linspace(0.001,0.9999,6)
    beta1s = np.linspace(0.9, 0.999, 5)
    beta2s = np.linspace(0.99,0.9999, 5)

    sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                eta=etas[6], lmd = lmds[0], epochs=100, minibatch_n = int(mbs[2]))
    
    sgd(sgd_method='vanilla')
    zvan_pred = (sgd.predict())
    print(f'sgd_method {score}')
    print('vanilla',score(z_test, act_func(X_test @ sgd.theta)))
    print('vanilla',score(z_train, zvan_pred))
    print('time', time.process_time() - start, 's')
    
    sgd.eta = etas[6]
    sgd.lmd = lmds[0]
    sgd.minibatch_n = int(mbs[2])
    sgd.gamma = gammas[2]
    sgd(sgd_method='momentum')
    zmom_pred = sgd.predict()
    print('momentum',score(z_test, act_func(X_test @ sgd.theta)))
    print('momentum',score(z_train, zmom_pred))

    print('time', time.process_time() - start, 's')

    sgd.minibatch_n = int(mbs[0])
    sgd.lmd = lmds[1]
    sgd.eta = etas[6]
    sgd.beta1 = beta1s[3]
    sgd.beta2 = beta2s[4]
    sgd(sgd_method='Adam')
    zAdam_pred = sgd.predict()
    print('Adam',score(z_test, act_func(X_test @ sgd.theta)))
    print('Adam',score(z_train, zAdam_pred))
    print('time', time.process_time() - start, 's')
        
def logistic():
    #np.random.seed(42)

    grad_meth = 'ordinary'
    act_func = Sigmoid()
    score = Accuracy()

    start = time.process_time()

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
                eta=etas[9], lmd = lmds[2], epochs=100, minibatch_n = int(mbs[3]))
    
    van_test, van_train, mom_test, mom_train, Adam_test, Adam_train \
    = [], [], [], [], [], []


    sgd(sgd_method='vanilla')
    print(f'sgd_method {score}')
    van_test.append(score(z_test, sgd.predict(X_test)))
    van_train.append(score(z_train, sgd.predict(X_train)))
    print('time', time.process_time() - start, 's')
    
    sgd.eta = etas[1]

    sgd.lmd = lmds[3]
    sgd.gamma = gammas[4] # best gamma in osp.py
    sgd.minibatch_n = int(mbs[2])
    sgd(sgd_method='momentum')
    mom_test.append(score(z_test, sgd.predict(X_test)))
    mom_train.append(score(z_train, sgd.predict(X_train)))

    print('time', time.process_time() - start, 's')

    sgd.eta = etas[2]
    sgd.lmd = lmds[0]
    sgd.minibatch_n = int(mbs[2])
    sgd.beta1 = beta1s[1]
    sgd.beta2 = beta2s[2]
    sgd(sgd_method='Adam')
    Adam_test.append(score(z_test, sgd.predict(X_test)))
    Adam_train.append(score(z_train, sgd.predict(X_train)))
    print('time', time.process_time() - start, 's')
  
    return np.array((van_test, van_train, mom_test, mom_train, Adam_test, Adam_train))

def skl():

    X_train, X_test, z_train, z_test = get_breastcancer()
    z_train = np.expand_dims(z_train, axis = 1)
    z_test = np.expand_dims(z_test, axis = 1)

    clf = LogisticRegression(penalty='none').fit(X_train,z_train)
    print(clf.score(X_test,z_test))

if __name__ == '__main__':
    #linear()
    a = []
    k = 100
    for i in range(k):
        a.append(logistic())
    
    # disable random seed!
    a = np.array(a).reshape(k,6)
    average = 100*np.mean(a,axis=0)
    van_test = average[0]
    van_train = average[1]
    mom_test = average[2]
    mom_train = average[3]
    Adam_test = average[4]
    Adam_train = average[5]
    print(van_test, van_train, mom_test, mom_train, Adam_test, Adam_train)

    skl()

    # 0.9556725146198821 0.9777889447236179 0.9584795321637418 0.9764070351758788 0.9538011695906424 0.9714070351758796
    # 91.38596491228061 94.62814070351754 94.7719298245613 96.8618090452261 95.39766081871335 97.25125628140701
    # 91.38596491228061 94.62814070351754 94.7719298245613 96.8618090452261 95.39766081871335 97.25125628140701