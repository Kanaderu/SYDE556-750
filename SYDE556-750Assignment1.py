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
		self.alpha=self.maxrate/(1 - self.xintercept)	#alpha=slope=(y2-y1)/(x2-x1)
		self.Jbias=self.maxrate-self.alpha
		self.rates=[]
		self.rates_noisy=[]

	def set_rates(self,x_vals):
		self.rates=[]
		for x in x_vals:
			J=np.maximum(0,self.alpha*self.e*x+self.Jbias)
			self.rates.append(np.maximum(0,J))
		return self.rates

	def set_rates_noisy(self,x_vals,noise):
		self.rates_noisy=[]
		for x in x_vals:
			eta=0
			J=np.maximum(0,self.alpha*self.e*x+self.Jbias)
			if noise !=0:
				eta=np.random.normal(loc=0,scale=noise)
			self.rates_noisy.append(np.maximum(0,J)+eta)
		return self.rates_noisy

	def get_rates(self):
		return self.rates

	def get_rates_noisy(self):
		return self.rates_noisy

class LIFneuron():

	def __init__(self,x_intercept,max_firing_rate,encoder,tau_ref,tau_rc):
		self.xintercept=x_intercept
		self.maxrate=max_firing_rate
		self.e=encoder
		self.rates=[]
		self.rates_noisy=[]
		self.tau_ref=tau_ref
		self.tau_rc=tau_rc
		# self.alpha=(np.exp((self.tau_ref*self.maxrate - 1)/(self.maxrate*self.tau_rc)))/((1-self.xintercept)*((1-self.tau_ref*self.maxrate - 1)/(self.maxrate*self.tau_rc)))
		self.alpha=(1-self.xintercept)**(-1)*((1-np.exp((self.tau_ref-self.maxrate**(-1))/self.tau_rc))**(-1)-1)
		self.Jbias=1-self.alpha*self.xintercept

	def set_rates(self,x_vals):
		self.rates=[]
		for x in x_vals:
			J=self.alpha*x*self.e+self.Jbias
			if J>1:
				rate=1/(self.tau_ref-self.tau_rc*np.log(1-1/J))
			else:
				rate=0
			self.rates.append(rate)
		return self.rates

	def set_rates_noisy(self,x_vals,noise):
		self.rates_noisy=[]
		for x in x_vals:
			eta=0
			J=self.alpha*x*self.e+self.Jbias
			if J>1:
				rate=1/(self.tau_ref -self.tau_rc*np.log(1-1/J))
			else:
				rate=0
			if noise !=0:
				eta=np.random.normal(loc=0,scale=noise)
			self.rates_noisy.append(rate+eta)
		return self.rates_noisy

	def get_rates(self):
		return self.rates

	def get_rates_noisy(self):
		return self.rates_noisy

def ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise):

	neurons=[]
	for i in range(n_neurons):
		n=ReLUneuron(x_intercept_array[i],max_rate_array[i],encoders[i])
		n.set_rates(x)
		n.set_rates_noisy(x,noise)
		neurons.append(n)
	return neurons

def LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise):

	neurons=[]
	for i in range(n_neurons):
		n=LIFneuron(x_intercept_array[i],max_rate_array[i],encoders[i],tau_ref,tau_rc)
		n.set_rates(x)
		n.set_rates_noisy(x,noise)
		neurons.append(n)
	return neurons

def neuron_responses(neurons,x,noise):

	fig=plt.figure()
	ax=fig.add_subplot(111)
	for n in neurons:
		if noise != 0:
			y=n.get_rates_noisy()
		else:
			y=n.get_rates()
		ax.plot(x,y)
	ax.set_xlim(-1,1)
	ax.set_xlabel('x')
	ax.set_ylabel('$a$ (Hz)')
	plt.show()

def get_optimal_decoders(neurons,x,S,noise):

	# Use A=matrix of activities (the firing of each neuron for each x value)
	# Noise is input in case the firing rates are noisy but optimization doesn't
	# account for that, as in 1.1d. For zero noise, noise=0 is input to the function.
	A_T=[]
	for n in neurons:
		if noise != 0:
			A_T.append(n.get_rates_noisy()) #should noise be added here, or only in estimation?
		else:
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

def get_optimal_decoders_noisy(neurons,x,S,noise):

	# Use A=matrix of activities (the firing of each neuron for each x value)
	A_T=[]
	for n in neurons:
		if noise != 0:
			A_T.append(n.get_rates_noisy()) #should noise be added here, or only in estimation?
		else:
			A_T.append(n.get_rates())
	A_T=np.matrix(A_T)
	A=np.transpose(A_T)
	x=np.transpose(np.matrix(x))
	upsilon=A_T*x/S
	gamma=A_T*A/S + np.identity(len(neurons))*noise**2
	d=np.linalg.inv(gamma)*upsilon
	
	return d

def get_state_estimate(neurons,x,d,noise):

	xhat=[]
	for j in range(len(x)):
		xhat_i=0
		for i in range(len(neurons)):
			if noise != 0:
				xhat_i+=float(d[i])*neurons[i].get_rates_noisy()[j]
			else:
				xhat_i+=float(d[i])*neurons[i].get_rates()[j]
		xhat.append(xhat_i)

	return xhat

def error_vs_neurons(N_list,min_fire_rate,max_fire_rate,min_x,max_x,x,noise_mag,averages,tau_ref,tau_rc):

	E_dist=[]
	E_noise=[]
	S=len(x)
	for n in N_list:
		n_neurons=n
		E_dist_n=[]
		E_noise_n=[]
		for a in range(averages):
			max_rate_array=np.random.uniform(100,200,n_neurons)
			x_intercept_array=np.random.uniform(-0.95,0.95,n_neurons)
			encoders=-1+2*np.random.randint(2,size=n_neurons)
			noise=noise_mag*np.max(max_rate_array)
			if n_type == 'ReLU':
				neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
			if n_type == 'LIF':
				neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
			d=get_optimal_decoders_noisy(neurons,x,S,noise)	#noisy optimization with noisy rates
			E_dist_n.append(get_e_dist(neurons,x,d,0))	#no noise
			E_noise_n.append(get_e_noise(d,noise))
		E_dist.append(np.average(E_dist_n))
		E_noise.append(np.average(E_noise_n))

	return E_dist,E_noise

def get_e_dist(neurons,x,d,noise):

	xhat=get_state_estimate(neurons,x,d,noise)
	E_dist=0.5*np.average(np.square(x-xhat))
	return E_dist

def get_e_noise(d,noise):
	E_noise=noise**2*np.sum(np.square(d))
	return E_noise




# ################################################################################################



def one_pt_one_a_thru_c(): #1.1a-c
	
	n_neurons=16
	min_fire_rate=100
	max_fire_rate=200
	min_x=-1
	max_x=1
	max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
	x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
	encoders=-1+2*np.random.randint(2,size=n_neurons)
	dx=0.05
	noise=0
	x=np.linspace(min_x,max_x,(max_x-min_x)/dx)

	neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
	neuron_responses(neurons,x,noise)

	S=len(x)
	d=get_optimal_decoders(neurons,x,S,noise)	#noiseless optimization with noiseless rates
	xhat=get_state_estimate(neurons,x,d,noise)	#noiseless rates
	
	fig=plt.figure()
	ax=fig.add_subplot(211)
	ax.plot(x,x,'b',label='$x$')
	ax.plot(x,xhat,'g',label='$\hat{x}$')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(212)
	ax.plot(x,x-xhat)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat)**2))],loc='best') 
	plt.show()
	# print 'RMSE', np.sqrt(np.average((x-xhat)**2))

def one_pt_one_d():	#1.1d

	n_neurons=16
	min_fire_rate=100
	max_fire_rate=200
	min_x=-1
	max_x=1
	max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
	x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
	encoders=-1+2*np.random.randint(2,size=n_neurons)
	dx=0.05
	x=np.linspace(min_x,max_x,(max_x-min_x)/dx)
	S=len(x)

	noise=0.2*np.max(max_rate_array)
	neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
	d=get_optimal_decoders(neurons,x,S,noise)	#noiseless optimization with noisy rates
	xhat=get_state_estimate(neurons,x,d,noise)	#noisy rates

	fig=plt.figure()
	ax=fig.add_subplot(211)
	ax.plot(x,x,'b',label='$x$')
	ax.plot(x,xhat,'g',label='$\hat{x}$')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(212)
	ax.plot(x,x-xhat)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat)**2))],loc='best') 
	plt.show()

def one_pt_one_e():	#1.1e
	
	n_neurons=16
	min_fire_rate=100
	max_fire_rate=200
	min_x=-1
	max_x=1
	max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
	x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
	encoders=-1+2*np.random.randint(2,size=n_neurons)
	dx=0.05
	noise=0
	x=np.linspace(min_x,max_x,(max_x-min_x)/dx)
	S=len(x)

	noise=0.2*np.max(max_rate_array)
	neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
	neuron_responses(neurons,x,noise)
	d1=get_optimal_decoders(neurons,x,S,noise)		#noiseless optimization with noisy rates
	d2=get_optimal_decoders_noisy(neurons,x,S,noise)	#noisy optimization with noisy rates
	xhat1=get_state_estimate(neurons,x,d1,noise)	#noisy rates
	xhat2=get_state_estimate(neurons,x,d2,noise)	#noisy rates

	fig=plt.figure()
	ax=fig.add_subplot(411)
	ax.plot(x,x,'b',label='$x$')
	ax.plot(x,xhat1,'g',label='$\hat{x}$')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(412)
	ax.plot(x,x-xhat1)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat1)**2))],loc='best')
	ax=fig.add_subplot(413)
	ax.plot(x,x,'b',label='$x$')
	ax.plot(x,xhat2,'g',label='$\hat{x}$')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(414)
	ax.plot(x,x-xhat2)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat2)**2))],loc='best') 
	plt.show()

def one_pt_two(): #1.2a-b

	N_list=[4,8,16,32,64,128,256,512]
	averages=5
	dx=0.05
	x=np.linspace(-1.0,1.0,2.0/dx)
	min_fire_rate=100
	max_fire_rate=200
	min_x=-0.95
	max_x=0.95

	noise_mag=0.1
	E_dist1,E_noise1 = get_error_estimates(N_list,min_fire_rate,max_fire_rate,min_x,max_x,x,noise_mag,averages,tau_ref,tau_rc,'ReLU')
	noise_mag=0.01
	E_dist2,E_noise2 = get_error_estimates(N_list,min_fire_rate,max_fire_rate,min_x,max_x,x,noise_mag,averages,tau_ref,tau_rc,'ReLU')

	fig=plt.figure()
	ax=fig.add_subplot(411)
	ax.loglog(N_list,E_dist1,'b',label='$E_{dist}$')
	ax.loglog(N_list,1./np.square(N_list),'g',label='$1/n^2$')
	ax.set_xlabel('$n_{neurons}$')
	ax.set_ylabel('$E_{dist}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(412)
	ax.loglog(N_list,E_noise1,label='$E_{noise}$')
	ax.loglog(N_list,1./np.array(N_list),'g',label='$1/n$')
	legend=ax.legend(loc='best',shadow=True)
	ax.set_xlabel('$n_{neurons}$')
	ax.set_ylabel('$E_{noise}$')
	ax=fig.add_subplot(413)
	ax.loglog(N_list,E_dist2,'b',label='$E_{dist}$')
	ax.loglog(N_list,1./np.square(N_list),'g',label='$1/n^2$')
	ax.set_xlabel('$n_{neurons}$')
	ax.set_ylabel('$E_{dist}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(414)
	ax.loglog(N_list,E_noise2,label='$E_{noise}$')
	ax.loglog(N_list,1./np.array(N_list),'g',label='$1/n$')
	legend=ax.legend(loc='best',shadow=True)
	ax.set_xlabel('$n_{neurons}$')
	ax.set_ylabel('$E_{noise}$')
	plt.show()

def one_pt_three():	#1.3

	n_neurons=16
	min_fire_rate=100
	max_fire_rate=200
	min_x=-1
	max_x=1
	max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
	x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
	encoders=-1+2*np.random.randint(2,size=n_neurons)
	dx=0.05
	x=np.linspace(min_x,max_x,(max_x-min_x)/dx)
	tau_ref=0.002
	tau_rc=0.02
	S=len(x)
	noise=0.0
	neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
	neuron_responses(neurons,x,noise)

	noise=0.2*np.max(max_rate_array)
	neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
	d1=get_optimal_decoders(neurons,x,S,noise)		#noiseless optimization with noisy rates
	d2=get_optimal_decoders_noisy(neurons,x,S,noise)	#noisy optimization with noisy rates
	xhat1=get_state_estimate(neurons,x,d1,noise)	#noisy rates
	xhat2=get_state_estimate(neurons,x,d2,noise)	#noisy rates

	fig=plt.figure()
	ax=fig.add_subplot(411)
	ax.plot(x,x,'b',label='$x$')
	ax.plot(x,xhat1,'g',label='$\hat{x}$')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(412)
	ax.plot(x,x-xhat1)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat1)**2))],loc='best')
	ax=fig.add_subplot(413)
	ax.plot(x,x,'b',label='$x$')
	ax.plot(x,xhat2,'g',label='$\hat{x}$')
	ax.set_ylim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(414)
	ax.plot(x,x-xhat2)
	ax.set_xlim(-1,1)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$x - \hat{x}$')
	legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat2)**2))],loc='best') 
	plt.show()

def main():

	one_pt_three()

main()