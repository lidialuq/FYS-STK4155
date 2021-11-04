import numpy as np


# see lecture notes week 39 for info
class SGD:
	def __init__(self, X, y, theta, eta, epochs):
		self.X = X # design matrix
		self.y = y # True parameters
		self.theta = theta # predicted parameters
		self.theta.shape
		self.eta = eta # learning rate
		self.epochs = epochs # number of epochs 

		# for SGD with momentum
		self.gamma = None # momentum parameter E [0,1]
		
		# for Adam
		self.beta1 = None # forgetting factor for gradient
		self.beta2 = None # forgetting factor for second moment of gradient
		
		
		self.N = int(len(self.y)) # number of parameters

		# idk really, from learning_schedule week 39 notes
		#self.t1 = self.epochs # end time 
		#self.t0 = self.t1/10 # start time
		
		

	def __call__(self, sgd_method, grad_method):
		if sgd_method == 'vanilla':
			self.__vanilla(grad_method)

		if sgd_method == 'momentum':
			self.__momentum(grad_method)

		if sgd_method == 'Adam':
			self.__Adam(grad_method)

	# time-based decay
	def __learning_schedule(self, epoch,eta):
		return eta/(1+epoch*self.eta/self.epochs)

	# used in notes week 39
	"""def __learning_schedule(self, t):
		return self.t0/(t+self.t1)
	"""

	def __gradient__(self, Xi, yi, method):
		if method == 'regular':
			gradient = 2 * Xi.T @ ((Xi @ self.theta) - yi)
		if method == 'ridge':
			gradient = 2 * (Xi.T @ ((Xi @ self.theta) - yi) + self.lmd*self.theta)
		return gradient

	# Vanilla SGD method, Newton's (-Raphson) method
	def __vanilla(self, grad_method):
		eta = self.eta
		for epoch in range(self.epochs):
			for i in range(self.N):
				rnd = np.random.randint(self.N)
				Xi = self.X[rnd:rnd+1]
				yi = self.y[rnd:rnd+1]
				gradients = self.__gradient__(Xi, yi, grad_method)
				
				eta = self.__learning_schedule(epoch,eta)
				self.theta = self.theta - eta*gradients

	def __momentum(self, grad_method):
		if self.gamma is None:
			raise Exception('Please initialize gamma as "class.gamma = value".')

		avg = 0
		eta = self.eta
		for epoch in range(self.epochs):
			for i in range(self.N):
				rnd = np.random.randint(self.N)
				Xi = self.X[rnd:rnd+1]
				yi = self.y[rnd:rnd+1]
				gradients = self.__gradient__(Xi, yi, grad_method)

				avg = avg*self.gamma + (1-self.gamma)*gradients
				eta = self.__learning_schedule(epoch,eta)
				self.theta = self.theta - eta*avg

	def __Adam(self, grad_method):
		m = 0
		v = 0
		eps = 1e-10 # some low value number to prevent division by zero
		for epoch in range(self.epochs):
			for i in range(self.N):
				rnd = np.random.randint(self.N)
				Xi = self.X[rnd:rnd+1]
				yi = self.y[rnd:rnd+1]
				gradients = self.__gradient__(Xi, yi, grad_method)

				m = m*self.beta1 + (1-self.beta1)*gradients
				mt = m / (1-self.beta1**(i+1))
				
				v = v*self.beta2 + (1-self.beta2)*(gradients**2)
				vt = v / (1-self.beta2**(i+1))

				self.theta = self.theta - self.eta * mt / (np.sqrt(vt) + eps)

	def predict(self):
		z = self.X @ self.theta
		return z

