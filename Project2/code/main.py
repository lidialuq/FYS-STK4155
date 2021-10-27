import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sgd import SGD
from data.franke_function import FrankeFunction
from helpers.cost_functions import MeanSquareError
from analysis_proj1 import LinearRegression


def main():
    np.random.seed(42)

    N = int(30)
    ff = FrankeFunction(axis_n = N, noise_var = 0, plot = False)

    X = ff.design_matrix(ff.x, ff.y, degree = 3)
    z = np.expand_dims(ff.z,axis=1)
    P = X.shape[1]

    th=np.random.rand(P,N)

    sgd = SGD(X, z, theta=th, eta=0.01, epochs=100)
    
    sgd(sgd_method='vanilla', grad_method='regular')
    zvan_pred = sgd.predict()
    print('sgd_method Mean Squared Error')
    print('vanilla',MeanSquareError()(z, zvan_pred))
    
    sgd.gamma = 0.001
    sgd(sgd_method='momentum', grad_method='regular')
    zmom_pred = sgd.predict()
    print('momentum',MeanSquareError()(z, zmom_pred))

    sgd.beta1 = 0.99
    sgd.beta2 = 0.999 
    sgd(sgd_method='Adam', grad_method='regular')
    zAdam_pred = sgd.predict() 
    print('Adam',MeanSquareError()(z, zAdam_pred))

    ols = LinearRegression(ff.x, ff.y, ff.z, degree = 3, split=True, test_size = 0.3, plot=False)
    print('ols',MeanSquareError()(ols.z_test, ols.z_model))


if __name__ == '__main__':
    main()