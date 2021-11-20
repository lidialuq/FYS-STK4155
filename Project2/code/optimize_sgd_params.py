# written by @ fredhof
# Grid search over SGD parameters, it's ugly, but works.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
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
        N = int(20)
        deg = 3
        X_train, X_test, z_train, z_test = get_ff_data(axis_n = N, degree = deg) # calls ff(axis_n = 20, noise_var = 0.1, plot = False)
    

    elif str(activation) == 'Sigmoid':
        act_func = Sigmoid()
        score = Accuracy()
        X_train, X_test, z_train, z_test = get_breastcancer()
        z_train = np.expand_dims(z_train, axis = 1)
        z_test = np.expand_dims(z_test, axis = 1)
        

    

    etas = np.logspace(-4,1,10)
    lmds = np.logspace(-6,1,1)
    
    if grad_meth == 'ridge':
        lmds = np.logspace(-6,1,5)

    mbs = np.linspace(1,20,5)
    
    # Grid search
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
                    values[num] = score(z_test, sgd.predict(X_test)), i, j, k
                    
                    num = num + 1

        name = f'{sgd_meth}_values_{activation}_{grad_meth}'
        np.save(name,values)
        return name
    
    gammas = np.linspace(0.001,0.9999,6)
    # best gamma
    gammas = np.array([gammas[4]])
    

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
                        values[num] = score(z_test, sgd.predict(X_test)), i, j, k, l

                        num = num + 1
        
        name = f'{sgd_meth}_values_{activation}_{grad_meth}'
        np.save(name,values)
        return name

    beta1s = np.linspace(0.9, 0.999, 5)
    # best beta1
    beta1s = np.array([beta1s[1]]) # log
    #beta1s = np.array([beta1s[2]]) # linear
    beta2s = np.linspace(0.99,0.9999, 5)
    # best beta2
    beta2s = np.array([beta2s[2]]) # log
    #beta2s = np.array([beta2s[4]]) # linear
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
                            values[num] = score(z_test, sgd.predict(X_test)), i, j, k, l, m

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


def find_vals(name, score, plot = False):
    file = np.load(name)
    if str(score) == 'Mean Squared Error':
        val = np.where(np.abs(file[:,0]) < 1, file[:,0], np.nan) # sets not wanted values to NaN
        idx = np.where(val == np.nanmin(val)) # finds the index with best values
    
    elif str(score) == 'Accuracy':
        val = np.where(np.abs(file[:,0]) > 1, np.nan, file[:,0])
        idx = np.where(val == np.nanmax(val))

    if plot and 'log' in name:
        etas = np.logspace(-4,1,10)
        lmds = np.logspace(-6,1,5)
      
        mbs = np.linspace(1,20,5)
        gammas = np.linspace(0.001,0.9999,6)
        beta1s = np.linspace(0.9, 0.999, 5)
        beta2s = np.linspace(0.99,0.9999, 5)
        


        # REDUCES file_arr TO ONLY BEST VALUES (if bestvalues are specified above this only runs over mbs)
        file_arr = np.copy(file)
        file_arr[:,0] = val
        idxs = [] # adding indexes in order mbs, gammas/beta1, beta2
        for i in range(3,file_arr.shape[1]):
            # finds the most occuring parameters, if there are more than one option
            count = np.bincount(file[idx,i].astype('int').ravel())
            count_idx = np.argmax(count)
            idxs.append(count_idx)
            
            best_idxs = np.where(file_arr[:,i] == count_idx)
            file_arr = file_arr[best_idxs]

        heat = np.zeros((5,10))

        # eta
        # as k//5 in loop

        # lmd
        j = 0
        for k in range(file_arr.shape[0]):
            heat[j,k//5] = file_arr[k,0]
            j += 1
            if j == 5:
                j = 0

        if str(score) == 'Mean Squared Error':
            rect_val = np.flip(np.unravel_index(np.nanargmin(heat,axis=None), heat.shape))
    
        elif str(score) == 'Accuracy':
            rect_val = np.flip(np.unravel_index(np.nanargmax(heat,axis=None), heat.shape))
    
        

        ax = sns.heatmap(heat, annot = True, fmt = '.4f', xticklabels=np.around(np.log10(etas),3), yticklabels=np.log10(lmds), vmax=np.nanmax(heat), vmin = np.nanmin(heat), cmap='viridis')
        ax.add_patch(Rectangle(rect_val,1,1,fill=False, edgecolor='b',lw=2))
        plt.xlabel(f'Learning rate ' +r'$ 10^\eta$')
        plt.ylabel(f'Regularization coefficient ' +r'$ 10^\lambda$')
       
        #plt.title(f'{name}' + f' minibatch_n = {int(mbs[idxs][0])}')

        plt.show()

    print('Done')
    print(f'File with name: {name}, and possible index parameters')
    print(f'{score}, eta_idx, lmd, minibatch_n, gamma or beta1, (beta2)')
    # IF BEST VALUES ENABLED WILL RETURN ALL BEST VALUES WITH INDEX 0
    return file[idx]



if __name__ == '__main__':
    #folder = 'data/optimization_sgd_files/' # run from 'code' directiory
    #folder2 = 'a/'
    #filename = folder+'momentum_values_Linear_ordinary'
    #files = os.listdir(folder)

    for filename in files:
        if 'Linear' in filename:
            idx = find_vals(f'{folder}{filename}', MeanSquareError(),plot = False)
        elif 'Sigmoid' in filename:
            idx = find_vals(f'{folder}{filename}', Accuracy(), plot = True)

    
    # ALL PROGRAMS, example
    #main(activation = Linear(), sgd_meth = 'vanilla', grad_meth = 'ridge')
    #main(activation = Sigmoid(), sgd_meth = 'vanilla', grad_meth = 'ridge')

    #main(activation = Linear(), sgd_meth = 'momentum', grad_meth = 'ridge')
    #main(activation = Sigmoid(), sgd_meth = 'momentum', grad_meth = 'ridge')

    #main(activation = Linear(), sgd_meth = 'Adam', grad_meth = 'ridge')
    #main(activation = Sigmoid(), sgd_meth = 'Adam', grad_meth = 'ridge')
    