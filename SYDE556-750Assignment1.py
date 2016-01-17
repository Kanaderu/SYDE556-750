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
		self.rates=[]

	def set_rates(self,J_array):
		self.rates=[]
		b=-self.xintercept*self.alpha	#y=mx+b  ==>  b=y-mx
		for J in J_array:
			self.rates.append(np.maximum(0,self.alpha*self.e*J+b))
		return self.rates

	def get_rates(self):
		return self.rates

	def Jbias(self):
		return self.xintercept

	def alpha(self):
		return self.alpha

def ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders):

	neurons=[]
	for i in range(n_neurons):
		n=ReLUneuron(x_intercept_array[i],max_rate_array[i],encoders[i])
		n.set_rates(x)
		neurons.append(n)
	return neurons

def ReLUresponses(neurons,x):

	fig=plt.figure()
	ax=fig.add_subplot(111)

	for n in neurons:
		y=n.get_rates()
		ax.plot(x,y)

	ax.set_xlim(-1,1)
	ax.set_xlabel('x')
	ax.set_ylabel('$a$ (Hz)')
	plt.show()

def get_optimal_decoders(neurons,x,S):

	# Use A=matrix of activities (the firing of each neuron for each x value)
	A_T=[]
	for n in neurons:
		A_T.append(n.get_rates())
	A_T=np.matrix(A_T)
	A=np.transpose(A_T)
	x=np.transpose(np.matrix(x))
	upsilon=A_T*x/S
	gamma=A_T*A/S
	d=np.linalg.inv(gamma)*upsilon

	# Brute force - doesn't work because gamma_ij=0 for some values of a_i*a_j,
	# I'm missing something...

	# d=[]
	# for i in range(len(neurons)):
	# 	d_i=0
	# 	for j in range(len(neurons)):
	# 		upsilon_j=0
	# 		gamma_ij=0
	# 		for k in range(len(x)):
	# 			upsilon_j+=neurons[j].get_rates()[k]*x[k]/S
	# 			gamma_ij+=neurons[j].get_rates()[k]*neurons[i].get_rates()[k]/S
	# 		d_i+=gamma_ij**(-1)*upsilon_j
	# 	d.append(d_i)

	return d

def get_state_estimate(neurons,x,d):

	xhat=[]
	for j in range(len(x)):
		xhat_i=0
		for i in range(len(neurons)):
			xhat_i+=float(d[i])*neurons[i].get_rates()[j]
		xhat.append(xhat_i)

	return xhat

def plot_error(x,xhat):

	fig=plt.figure()
	ax=fig.add_subplot(211)
	ax.plot(x,x,'b')
	ax.plot(x,xhat,'g')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	ax=fig.add_subplot(212)
	ax.plot(x,x-xhat)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	plt.show()
	print 'RMSE'
	print np.sqrt(np.average((x-xhat)**2))

def main():

	#Q1 setup
	n_neurons=16
	max_rate_array=np.random.uniform(100,200,n_neurons)
	x_intercept_array=np.random.uniform(-0.95,0.95,n_neurons)
	encoders=-1+2*np.random.randint(2,size=n_neurons)
	dx=0.05
	x=np.linspace(-1.0,1.0,2.0/dx)
	
	#1.1a
	neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders)
	# ReLUresponses(neurons,x)

	#1.1b
	S=len(x)
	d=get_optimal_decoders(neurons,x,S)

	#1.1c
	xhat=get_state_estimate(neurons,x,d)
	plot_error(x,xhat)


main()