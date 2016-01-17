# Peter Duggins
# SYDE 556/750
# Jan 25, 2015
# Assignment 1

import numpy as np
import matplotlib.pyplot as plt

class ReLUneuron():

	def __init__(self,x_intercept,max_firing_rate,encoder):
		self.xintercept=x_intercept
		self.maxrate=max_firing_rate
		self.e=encoder
		self.alpha=(self.maxrate - 0.0)/(1.0 - self.xintercept)	#alpha=slope=(y2-y1)/(x2-x1)
		self.b=-self.xintercept*self.alpha	#y=mx+b  ==>  b=y-mx

	def rate(self,J):
		rate=self.alpha*self.e*J+self.b	
		return np.maximum(0,rate)

	def rates(self,J_array):
		rates=[]
		for J in J_array:
			rates.append(np.maximum(0,self.alpha*self.e*J+self.b))
		return rates

	def Jbias(self):
		return self.xintercept

	def alpha(self):
		return self.alpha

def ReLUresponses(n_neurons,x_intercept_array,max_rate_array,x,encoders):

	fig=plt.figure()
	ax=fig.add_subplot(111)

	for i in range(n_neurons):
		n=ReLUneuron(x_intercept_array[i],max_rate_array[i],encoders[i])
		y=n.rates(x)
		ax.plot(x,y)

	ax.set_xlim(-1,1)
	ax.set_xlabel('x')
	ax.set_ylabel('$a$ (Hz)')
	plt.show()


def main():

	n_neurons=4
	max_rate_array=np.random.uniform(100,200,n_neurons)
	x_intercept_array=np.random.uniform(-0.95,0.95,n_neurons)
	encoders=-1+2*np.random.randint(2,size=n_neurons)
	dx=0.05
	x=np.linspace(-1.0,1.0,2.0/dx)
	ReLUresponses(n_neurons,x_intercept_array,max_rate_array,x,encoders)




main()