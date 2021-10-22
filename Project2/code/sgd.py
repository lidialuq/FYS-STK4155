import numpy as np


# see lecture notes week 39 for info
class SGD:
	def __init__(self, X, y, theta, eta, gamma, epochs):
		self.X = X
		self.y = y
		self.theta = theta
		self.eta = eta # learning rate
		self.gamma = gamma # momentum parameter E [0,1]
		self.epochs = epochs
		self.t1 = self.epochs #
		self.t0 = self.t1/10

	def __call__(self, method):
		if method == 'vanilla':
			self.__vanilla()

		if method == 'momentum':
			self.__momentum()

		if method == 'Adam':
			self.__Adam()

	def __learning_schedule(self, t):
		return self.t0/(t+self.t1)


	# Vanilla SGD method, Newton's (-Raphson) method
	def __vanilla(self):
		for epoch in range(self.epochs):
			for i in range(len(self.y)):
				rnd = np.random.randint(len(self.y))
				Xi = self.X[rnd:rnd+1]
				yi = self.y[rnd:rnd+1]
				grads = 2*Xi.T @ ((Xi @ self.theta) - yi)
				eta = self.__learning_schedule(epoch*len(self.y)+i)
				self.theta -= self.eta*grads

	def __momentum(self):
		avg = 0

		for epoch in range(self.epochs):
			for i in range(len(self.y)):
				rnd = np.random.randint(len(self.y))
				Xi = self.X[rnd:rnd+1]
				yi = self.y[rnd:rnd+1]
				grads = 2*Xi.T @ ((Xi @ self.theta) - yi)
				avg = avg*self.gamma + (1-self.gamma)*grads
				eta = self.__learning_schedule(epoch*len(self.y)+i)
				self.theta -= self.eta*avg

	def __Adam(self):
		return		


