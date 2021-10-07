# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 21:58:52 2021

@author: lidia
"""
from analysis import FrankeFunction, LinearRegression
import matplotlib.pyplot as plt
import numpy as np
#Get dir for saving plots
from os.path import dirname, abspath, join
parent_dir = dirname(abspath(__file__))
plots_dir = join(parent_dir, 'plots')
print('PLot dir: {}'.format(plots_dir))


def plot_conf_interval(data,model='OLS',lamb=0,noise_var=0.05,axis_n=10,degree=5,
                       save = False):
    '''
    Exercice 1
    Plot betas with confidence intervals. Get MSE and R2
    '''
    np.random.seed(seed=42)
    reg = LinearRegression(data[0], data[1], data[2], degree = degree, split=False,
                           model=model,lamb=lamb)
    beta_var = reg.beta_variance(reg.X, sigma_sq = noise_var)
    x = list(range(0, len(beta_var)))
    beta_conf = 1.96*np.sqrt(beta_var/axis_n**2)
    
    MSE = reg.MSE(reg.z_data, reg.z_model)
    R2 = reg.R2(reg.z_data, reg.z_model)
    
    plt.errorbar(x, reg.beta, yerr=beta_conf, markersize=4, linewidth=1, \
                 ecolor="black", fmt='o', capsize=5, capthick=1)
    plt.title("Regression parameters")
    x_labels = [r"$\beta_"+"{{{:}}}$".format(i) for i in range(len(reg.beta))]
    plt.xticks(x, x_labels)
    plt.show()
    
    if save: 
        plt.savefig(join(plots_dir, 'b_conf-n={}-noise={}.pdf'.format(axis_n**2, noise_var)))
    
    print("MSE = {}".format(MSE))
    print("R2 = {}".format(R2))
    
    

def plot_train_test_MSE(data,model='OLS',lamb=0, max_degree=9, axis_n = 1, bootstrap_nr = 100, 
                        noise_var = 'unknown', save = False):
    '''
    Exercice 2
    Plot test and train MSE. For "nice" plot, axis_n=50, noise_var=0.05, 
    # deg range(1,9)
    '''
    np.random.seed(seed=42)
    trainMSE = []
    testMSE = []
    x = []
    for deg in range(1,max_degree):
        x.append(deg)
        reg = LinearRegression(data[0], data[1], data[2], degree = deg, split=True,
                           model=model,lamb=lamb)
        
#        MSE_, _, _, MSE_train= reg.bootstrap(reg.X_train, reg.X_test, 
#                                   reg.z_train, reg.z_test,
#                                   bootstrap_nr=bootstrap_nr)

        trainMSE.append( reg.MSE(reg.z_train, reg.z_model_train))
        testMSE.append( reg.MSE(reg.z_test, reg.z_model))
#        testMSE.append( MSE_)    
#        trainMSE.append( MSE_train)
        
    plt.plot(x, testMSE, 'b') 
    plt.plot(x, trainMSE, 'r') 
    plt.legend(["Test MSE", "Train MSE"])
    plt.title("Model MSE")
    plt.xlabel("Model complexity (degree)")
    plt.ylabel("MSE")
    plt.show()
    
    if save: 
        plt.savefig(join(plots_dir, 'MSEtest_train-n={}-noise={}.pdf'.format(axis_n**2, noise_var)))

    
def plot_bias_var_bootstrap(data,model='OLS',lamb=0,bootstrap_nr=100,max_degree=9,axis_n = 1,
                            noise_var = 'unknown', save = False):
    '''
    Plot the mean of the bias, variance and MSE over samples of the data for 
    The sampling is done using the bootstrap method. 
    '''
    np.random.seed(seed=42)
    MSE, bias, var, bias_plus_var = [],[],[],[]
    x = []
    
    for deg in range(1,max_degree):
        x.append(str(deg))  # make xaxis ticks integers
        reg2 = LinearRegression(data[0], data[1], data[2], degree = deg, split=True,
                           model=model,lamb=lamb,test_size = 0.3)
        MSE_, bias_, var_, _ = reg2.bootstrap(reg2.X_train, reg2.X_test, 
                                          reg2.z_train, reg2.z_test,
                                          bootstrap_nr=bootstrap_nr)
        MSE.append(MSE_)
        bias.append(bias_)
        var.append(var_) 
        bias_plus_var.append(bias_ + var_)
    
    
#    print('MSE: {}\n'.format(MSE))
#    print('bias: {}\n'.format(bias))
#    print('var: {}\n'.format(var))
#    print('bias+var: {}\n'.format(bias_plus_var))
    plt.plot(x, MSE, 'b') 
    plt.plot(x, bias, 'r') 
    plt.plot(x, var, 'g') 
    # plt.yscale('log')
    plt.legend(["MSE", "bias", "var"])
    plt.title("Bias-variance (bootstrap)")
    plt.xlabel("Model complexity (degree)")
    plt.show()
    if save:
        plt.savefig(join(plots_dir, 'biasvar-n={}-noise={}.pdf'.format(axis_n**2, noise_var)))

     
def plot_mse_bootstrap(data,model='OLS',lamb=0,bootstrap_nr=100,max_degree=9,axis_n = 1,
                            noise_var = 'unknown', save = False):
    '''
    Plot the test and train MSE over samples of the data for bootstrap sampling
    The sampling is done using the bootstrap method. 
    '''
    np.random.seed(seed=42)
    MSE, MSE_train = [],[]
    x = []
    
    for deg in range(1,max_degree):
        x.append(str(deg))  # make xaxis ticks integers
        reg2 = LinearRegression(data[0], data[1], data[2], degree = deg, split=True,
                           model=model,lamb=lamb,test_size = 0.3)
        MSE_, _, _ , MSE_train_ = reg2.bootstrap(reg2.X_train, reg2.X_test, 
                                          reg2.z_train, reg2.z_test,
                                          bootstrap_nr=bootstrap_nr)
        MSE.append(MSE_)
        MSE_train.append(MSE_train_)

    plt.plot(x, MSE, 'b') 
    plt.plot(x, MSE_train, 'r') 
    plt.legend(["Test MSE", "Train MSE", "var"])
    plt.title("MSE (bootstrap)")
    plt.xlabel("Model complexity (degree)")
    plt.show()
    if save:
        plt.savefig(join(plots_dir, 'mse_bootstrap-n={}-noise={}.pdf'.format(axis_n**2, noise_var)))       
        

def plot_mse_kfolds(data,model='OLS',lamb=0,kfld=5,max_degree=9, save=False):
    np.random.seed(seed=42)
    MSE, MSE_train = [],[]
    x = []
    
    for deg in range(1,max_degree):
        x.append(deg)
        reg = LinearRegression(data[0], data[1], data[2], degree = deg, split=True,
                           model=model,lamb=lamb)
        X_des = reg.design_matrix(x=data[0],y=data[1])
        a = LinearRegression(data[0], data[1], data[2],split=True).split_and_scale(X_des, data[2],test_size=0.3)
        MSE_, MSE_train_ = reg.kfold(X_des, data[2],k=kfld)
        MSE.append(MSE_)
        MSE_train.append(MSE_train_)

        
    # plt.yscale('log')
    plt.plot(x, MSE, 'b') 
    plt.plot(x, MSE_train, 'r') 
    plt.legend(["Test MSE", "Train MSE"])
    plt.title("MSE (k-folds)")
    plt.xlabel("Model complexity (degree)")
    plt.ylabel("MSE")
    if save:
        plt.savefig(join(plots_dir, 'mse_kfold-n={}-noise={}.pdf'.format(axis_n**2, noise_var)))     


#%%Exercise 1
axis_n = 15
noise_var = 0.05
save = False
ff = FrankeFunction(axis_n = axis_n, noise_var = noise_var, plot=False)
ff_data = [ff.x,ff.y,ff.z]

plot_conf_interval(ff_data, noise_var=noise_var ,axis_n=axis_n) 


#%%Exercise 2
plot_train_test_MSE(ff_data, axis_n = axis_n, noise_var = noise_var, max_degree = 10)

plot_bias_var_bootstrap(ff_data, max_degree=10, bootstrap_nr = 50)


#%%Exercise 3
plot_mse_bootstrap(ff_data, max_degree=12, bootstrap_nr = 500)

plot_mse_kfolds(ff_data, kfld=10, max_degree=12)
'''
#%%Exercise 4
#MSE
n_lamb = 8
lambs = np.logspace(-7,0,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_train_test_MSE(ff_data,model='Ridge',lamb=lambs[i],max_degree=15)
    plt.title('$\lambda$='+str(lambs[i]))

plt.tight_layout()

#Bootstrap
n_lamb = 8
lambs = np.logspace(-7,0,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_bootstrap(ff_data,model='Ridge',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))

plt.tight_layout()

#K-folds
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_kfolds(ff_data,model='Ridge',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))

plt.tight_layout()

#%%Exercise 5

n_lamb = 8
lambs = np.logspace(-7,0,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_train_test_MSE(ff_data,model='Lasso',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))

plt.tight_layout()

#Bootstrap
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_bootstrap(ff_data,model='Lasso',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))

plt.tight_layout()


#K-folds
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_kfolds(ff_data,model='Lasso',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))

plt.tight_layout()

#%%Exercise 6
from imageio import imread
# Load the terrain
N = 100
terrain1 = imread('E:\\UiO\\Courses\\Machine_learning\\SRTM_data_Norway_1.tif')
terrain = terrain1[:N,:N]# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


x = np.linspace(0,1,len(terrain))
y = np.linspace(0,1,len(terrain.T))
x_grid, y_grid = np.meshgrid(x,y)
terrain_data = [x_grid.flatten(),y_grid.flatten(),terrain.flatten()]

reg = LinearRegression(terrain_data[0], terrain_data[1], terrain_data[2], degree = 9, 
                       split=False, model='OLS',lamb=0)

#%%
#Terrain data OLS
plt.figure()
plot_train_test_MSE(terrain_data)
# plot_conf_interval(terrain_data)
plot_bias_var_bootstrap(terrain_data)
plot_bias_var_kfolds(terrain_data)
#%%    
#Terrain data Ridge
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_train_test_MSE(terrain_data,model='Ridge',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))
plt.tight_layout()

#Bootstrap
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_bootstrap(terrain_data,model='Ridge',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))
plt.tight_layout()

n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_kfolds(terrain_data,model='Ridge',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))
plt.tight_layout()

#%%
#Terrain data Lasso
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_train_test_MSE(terrain_data,model='Lasso',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))
plt.tight_layout()

#Bootstrap
n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_bootstrap(terrain_data,model='Lasso',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))
plt.tight_layout()

n_lamb = 8
lambs = np.logspace(-4,3,n_lamb)
plt.figure()
for i in range(n_lamb):
    plt.subplot(2,4,i+1)
    plot_bias_var_kfolds(terrain_data,model='Lasso',lamb=lambs[i])
    plt.title('$\lambda$='+str(lambs[i]))
plt.tight_layout()
'''