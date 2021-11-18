import numpy as np

from helpers.cost_functions import BinaryCrossEntropy
from helpers.activations import Sigmoid


# see lecture notes week 39 for info
class SGD:
	def __init__(self, activation, grad_method, X, z, eta, lmd, epochs, minibatch_n, tol = 1e-7):
		self.activation = activation
		self.grad_method = grad_method
		self.X = X # design matrix
		self.z = z # True parameters
		self.eta = eta # learning rate
		self.lmd = lmd # Ridge hyperparameter
		self.epochs = epochs # number of epochs 
		self.minibatch_n = minibatch_n # number of mini-batches, size is self.N//self.minibatch_n 
		# where 0 = DG, 1 = SGD, N = SGD with mini-batches of size N
		self.tol = tol # tolerance for cost-functions, and Adam
		if str(activation) == 'Linear':
			self.tol = self.tol * 1e-2
		# if self.tol < 1e-9, Adam takes a lot more time
		self.theta = None # predicted parameters

		# for SGD with momentum
		self.gamma = None # exponential decay factor E [0,1], 
		# determines the relative contribution of the current gradient and earlier gradients to the weight change
		
		# for Adam
		self.beta1 = None # forgetting factor for gradient
		self.beta2 = None # forgetting factor for second moment of gradient
		
		
		self.N = X.shape[0] 
		self.P = X.shape[1] # number of parameters/lenght of self.theta
		

	def __call__(self, sgd_method):
		if sgd_method == 'vanilla':
			self.__vanilla()

		if sgd_method == 'momentum':
			self.__momentum()

		if sgd_method == 'Adam':
			self.__Adam()

	def __init_th(self):
		#np.random.seed(42) # easy way to guarantee same values for self.theta,
		# NECCESSARY IF RUNNING 'optimize_sgd_params.py', it resets the seed, so you get the same theta values
		return np.random.normal(0,1, size=(self.P,1))

	# time-based decay
	def __learning_schedule(self, epoch, eta):
		return eta/(1+epoch*self.eta/self.epochs)

	# linear decay, comment other __learning_schedule and uncomment if switch
	#def __learning_schedule(self, t, eta):
	#	return eta - self.eta/(self.epochs*self.N)
	
	def __convergence_cost__(self):
		if self.grad_method == 'ordinary':
			cost = 1/self.N * (self.activation(self.X @ self.theta) - self.z).T @ (self.activation(self.X @ self.theta) - self.z)
		elif self.grad_method == 'ridge':
			L2 = self.lmd/self.N * np.sum(self.theta**2)
			cost = 1/self.N * (self.activation(self.X @ self.theta) - self.z).T @ (self.activation(self.X @ self.theta) - self.z) + L2
		return np.mean(cost)

	def __gradient__(self):
		rnd = self.minibatch_n*np.random.randint(self.N//self.minibatch_n)
		Xi = self.X[rnd:rnd+self.minibatch_n] # mbn = 1 -> SGD, mbn > 1 -> SGD with minibatches
		zi = self.z[rnd:rnd+self.minibatch_n]

		if self.grad_method == 'ordinary':
			gradient = 2 * Xi.T @ (self.activation(Xi @ self.theta) - zi)
		elif self.grad_method == 'ridge':
			gradient = 2 * (Xi.T @ (self.activation(Xi @ self.theta) - zi) + self.lmd*self.theta)
		return gradient

	# Vanilla SGD method, Newton's (-Raphson) method
	def __vanilla(self):
		eta = self.eta
		fin = False
		cost0 = 0
		self.theta = self.__init_th()
		for epoch in range(self.epochs):
			cost = np.zeros(self.N)
			for i in range(self.N):
				gradients = self.__gradient__()
				cost1 = self.__convergence_cost__()
				cost[i] = cost1

				if np.abs(cost0 - cost1) < self.tol or np.isnan(np.abs(cost0 - cost1)):
					fin = True
					break
				cost0 = cost1

				eta = self.__learning_schedule(epoch,eta)
				self.theta = self.theta - eta*gradients
			
			#print(f'Epoch {epoch}, average train loss {np.mean(cost)}')
			if fin:
				#print(f'Theta reached at epoch {epoch}. ')
				break

	def __momentum(self):
		if self.gamma is None:
			raise Exception('Please initialize gamma as "class.gamma = value".')

		alpha = 0
		cost0 = 0
		fin = False
		eta = self.eta
		self.theta = self.__init_th()
		for epoch in range(self.epochs):
			cost = np.zeros(self.N)
			for i in range(self.N):
				gradients = self.__gradient__()
				cost1 = self.__convergence_cost__()
				cost[i] = cost1

				if np.abs(cost0 - cost1) < self.tol or np.isnan(np.abs(cost0 - cost1)):
					fin = True
					break
				cost0 = cost1

				eta = self.__learning_schedule(epoch,eta)

				alpha = alpha*self.gamma + eta*gradients
				
				self.theta = self.theta - alpha
			#print(f'Epoch {epoch}, average train loss {np.mean(cost)}')
			if fin:
				#print(f'Theta reached at epoch {epoch}. ')
				break

	def __Adam(self):
		if self.beta1 is None or self.beta2 is None:
			raise Exception('Please initialize beta1/beta2 as "class.beta = value".')

		m = 0
		v = 0
		cost0 = 0
		fin = False
		self.theta = self.__init_th()
		# self.tol - some low value number to prevent division by zero, and in cost, see __init__
		for epoch in range(self.epochs):
			cost = np.zeros(self.N)
			for i in range(self.N):
				gradients = self.__gradient__()
				cost1 = self.__convergence_cost__()
				cost[i] = cost1
				if np.abs(cost0 - cost1) < self.tol or np.isnan(np.abs(cost0 - cost1)):
					fin = True
					break
				cost0 = cost1

				m = m*self.beta1 + (1-self.beta1)*gradients
				mt = m / (1-self.beta1**(i+1))
				
				v = v*self.beta2 + (1-self.beta2)*(gradients**2)
				vt = v / (1-self.beta2**(i+1))

				self.theta = self.theta - self.eta * mt / (np.sqrt(vt) + self.tol)
			#print(f'Epoch {epoch}, average train loss {np.mean(cost)}')
			if fin:
				#print(f'Theta reached at epoch {epoch}. ')
				break

	def predict(self, X):
		z = self.activation(X @ self.theta)
		return z

