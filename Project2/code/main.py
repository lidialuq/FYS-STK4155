import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sgd import SGD



def main():
	np.random.seed(8)
	a = np.pi # creating some random function
	b = -np.e
	c = 42
	x = np.random.randint(-5,5, size = 100)
	y = a*x*x + b*x + c
	
	X = np.c_[np.ones((100,1)),x,x*x]

	sgd = SGD(X,y, theta=np.random.rand(3,1), eta=0.001, gamma = 0.1, epochs=100)
	
	sgd(method='vanilla')
	print(sgd.theta)

	sgd(method='momentum')
	print(sgd.theta)

	#sgd(method='Adam')
	#print(sgd.theta)



if __name__ == '__main__':
	main()