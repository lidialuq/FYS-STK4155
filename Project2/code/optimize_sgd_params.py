import numpy as np
import os

from sgd import SGD
from data.franke_function import FrankeFunction, get_ff_data
from helpers.cost_functions import MeanSquareError, Accuracy
from helpers.activations import Sigmoid, Linear

from data.breast_cancer import get_breastcancer

def main(activation, sgd_meth, grad_meth, SEED = 42):
    np.random.seed(SEED)
    print(f'Running simulations with activation = {activation},\
     sgd_method = {sgd_meth}, grad_method = {grad_meth} with np.random.seed = {SEED}')
    
    if str(activation) == 'Linear':
        act_func = Linear()
        score = MeanSquareError()

    elif str(activation) == 'Sigmoid':
        act_func = Sigmoid()
        score = Accuracy()
    
    N = int(20)
    deg = 3

    #X_train, X_test, z_train, z_test = get_ff_data(axis_n = N, degree = deg) # calls ff(axis_n = 20, noise_var = 0.1, plot = False)
    X_train, X_test, z_train, z_test = get_breastcancer()
    z_train = np.expand_dims(z_train, axis = 1)
    z_test = np.expand_dims(z_test, axis = 1)

    etas = np.logspace(-4,1,10)
    lmds = np.logspace(-6,1,1)
    if grad_meth == 'ridge':
        lmds = np.logspace(-6,1,5)

    mbs = np.linspace(0,20,5)
    

    def vanilla(etas, lmds):
        values = np.empty(shape=(etas.shape[0]*lmds.shape[0]*mbs.shape[0],4))
        num = 0
        sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                                    eta=etas[0], lmd = lmds[0], epochs=100, minibatch_n = int(mbs[0]))

        for i in range(len(etas)):
            for j in range(len(lmds)):
                for k in range(len(mbs)):
                    sgd.eta = etas[i]
                    sgd.lmd = lmds[j]
                    sgd.minibatch_n = int(mbs[k])

                    sgd(sgd_method='vanilla')
                    zvan_pred = sgd.predict()
                    values[num] = score(z_train, zvan_pred), i, j, k
                    
                    num = num + 1

        name = f'{sgd_meth}_values_{activation}_{grad_meth}'
        np.save(name,values)
        return name
    
    gammas = np.linspace(0.001,0.9999,6)

    def momentum(etas, lmds, gammas):
        values = np.empty(shape=(etas.shape[0]*lmds.shape[0]*mbs.shape[0]*gammas.shape[0],5))
        num = 0
        sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                                    eta=etas[0], lmd = lmds[0], epochs=100, minibatch_n = int(mbs[0]))

        for i in range(len(etas)):
            for j in range(len(lmds)):
                for k in range(len(mbs)):
                    for l in range(len(gammas)):
                        sgd.eta = etas[i]
                        sgd.lmd = lmds[j]
                        sgd.minibatch_n = int(mbs[k])
                        sgd.gamma = gammas[l]

                        sgd(sgd_method='momentum')
                        zvan_pred = sgd.predict()
                        values[num] = score(z_train, zvan_pred), i, j, k, l

                        num = num + 1
        
        name = f'{sgd_meth}_values_{activation}_{grad_meth}'
        np.save(name,values)
        return name

    beta1s = np.linspace(0.9, 0.999, 5)
    beta2s = np.linspace(0.99,0.9999, 5)

    def Adam(etas, lmds, beta1s, beta2s):
        values = np.empty(shape=(etas.shape[0]*lmds.shape[0]*mbs.shape[0]*beta1s.shape[0]*beta2s.shape[0], 6))
        num = 0
        sgd = SGD(activation = act_func, grad_method = grad_meth, X=X_train, z=z_train, \
                                    eta=etas[0], lmd = lmds[0], epochs=100, minibatch_n = int(mbs[0]))

        for i in range(len(etas)):
            for j in range(len(lmds)):
                for k in range(len(mbs)):
                    for l in range(len(beta1s)):
                        for m in range(len(beta2s)):
                            sgd.eta = etas[i]
                            sgd.lmd = lmds[j]
                            sgd.minibatch_n = int(mbs[k])               
                            sgd.beta1 = beta1s[l]
                            sgd.beta2 = beta2s[m]

                            sgd(sgd_method='Adam')
                            zvan_pred = sgd.predict()
                            values[num] = score(z_train, zvan_pred), i, j, k, l, m

                            num = num + 1
        
        name = f'{sgd_meth}_values_{activation}_{grad_meth}'
        np.save(name,values)
        return name

    if sgd_meth == 'vanilla':
        name = vanilla(etas, lmds)
    elif sgd_meth == 'momentum':
        name = momentum(etas, lmds, gammas)
    elif sgd_meth == 'Adam':
        name = Adam(etas, lmds, beta1s, beta2s)
    print('Done')

    return name


def find_vals(name, score):
    file = np.load(name)

    if str(score) == 'Mean Squared Error':
        val_min = np.where(np.abs(file[:,0]) < 1, file[:,0], 2)
        idx = np.where(val_min == np.min(val_min))
    
    elif str(score) == 'Accuracy':
        val_max = np.where(np.abs(file[:,0]) > 1, -2, file[:,0])
        idx = np.argwhere(val_max == np.nanmax(val_max))

    print('Done')
    print(f'File with name: {name}, and possible index parameters')
    print(f'{score}, eta_idx, lmd, minibatch_n, gamma or beta1, (beta2)')
    return file[idx]


if __name__ == '__main__':
    #filename = main(activation = Linear(), sgd_meth = 'momentum', grad_meth = 'ordinary')
    """folder = 'data/optimization_sgd_files/' # run from 'code' directiory
    #filename = folder+'momentum_values_Linear_ordinary'
    files = os.listdir(folder)

    for filename in files:
        if 'Linear' in filename:
            idx = find_vals(f'{folder}{filename}', MeanSquareError())
        elif 'Sigmoid' in filename:
            idx = find_vals(f'{folder}{filename}', Accuracy())
        print(idx)
    """
    # ALL FILES
    #main(activation = Linear(), sgd_meth = 'vanilla', grad_meth = 'ordinary')
    #main(activation = Linear(), sgd_meth = 'vanilla', grad_meth = 'ridge')
    #main(activation = Sigmoid(), sgd_meth = 'vanilla', grad_meth = 'ordinary')
    #main(activation = Sigmoid(), sgd_meth = 'vanilla', grad_meth = 'ridge')

    #main(activation = Linear(), sgd_meth = 'momentum', grad_meth = 'ordinary')
    #main(activation = Linear(), sgd_meth = 'momentum', grad_meth = 'ridge')
    main(activation = Sigmoid(), sgd_meth = 'momentum', grad_meth = 'ordinary')
    main(activation = Sigmoid(), sgd_meth = 'momentum', grad_meth = 'ridge')

    #main(activation = Linear(), sgd_meth = 'Adam', grad_meth = 'ordinary')
    #main(activation = Linear(), sgd_meth = 'Adam', grad_meth = 'ridge')
    main(activation = Sigmoid(), sgd_meth = 'Adam', grad_meth = 'ordinary')
    main(activation = Sigmoid(), sgd_meth = 'Adam', grad_meth = 'ridge')
    