# Peter Duggins
# SYDE 556/750
# March 3, 2015
# Assignment 3

import numpy as np
import matplotlib.pyplot as plt

class LIFneuron():

    def __init__(self,x_intercept,max_firing_rate,encoder,tau_ref,tau_rc):
        self.xintercept=x_intercept
        self.maxrate=max_firing_rate
        self.e=encoder
        self.tau_ref=tau_ref
        self.tau_rc=tau_rc
        self.alpha=(1-np.dot(self.xintercept,self.e))**(-1)*((1-np.exp((self.tau_ref-self.maxrate**(-1))/self.tau_rc))**(-1)-1)
        self.Jbias=1-self.alpha*np.dot(self.xintercept,self.e)
        self.sample_rates=[]
        self.sample_rates_noisy=[]
        self.custom_rates=[]
        self.custom_rates_noisy=[]
        self.custom_x=np.array([])

    def set_sample_rates(self,x_vals):
        self.sample_x=x_vals
        self.sample_rates=[]
        for x in x_vals:
            J=self.alpha*np.dot(x,self.e)+self.Jbias
            if J>1:
                rate=1/(self.tau_ref-self.tau_rc*np.log(1-1/J))
            else:
                rate=0
            self.sample_rates.append(float(rate))
        return self.sample_rates

    def set_sample_rates_noisy(self,x_vals,noise):
        self.sample_x=x_vals
        self.sample_rates_noisy=[]
        for x in x_vals:
            eta=0
            J=self.alpha*np.dot(x,self.e)+self.Jbias
            if J>1:
                rate=1/(self.tau_ref -self.tau_rc*np.log(1-1/J))
            else:
                rate=0
            if noise !=0:
                eta=np.random.normal(loc=0,scale=noise)
            self.sample_rates_noisy.append(float(rate+eta))
        return self.sample_rates_noisy

    def get_sample_rates(self):
        return self.sample_rates

    def get_sample_rates_noisy(self):
        return self.sample_rates_noisy

    def set_custom_rates(self,x_vals):
        self.custom_x=x_vals
        self.custom_rates=[]
        for x in x_vals:
            J=self.alpha*np.dot(x,self.e)+self.Jbias
            if J>1:
                rate=1/(self.tau_ref-self.tau_rc*np.log(1-1/J))
            else:
                rate=0
            self.custom_rates.append(float(rate))
        return self.custom_rates

    def set_custom_rates_noisy(self,x_vals,noise):
        self.custom_x=x_vals
        self.custom__rates_noisy=[]
        for x in x_vals:
            eta=0
            J=self.alpha*np.dot(x,self.e)+self.Jbias
            if J>1:
                rate=1/(self.tau_ref -self.tau_rc*np.log(1-1/J))
            else:
                rate=0
            if noise !=0:
                eta=np.random.normal(loc=0,scale=noise)
            self.custom_rates_noisy.append(float(rate+eta))
        return self.custom_rates_noisy

    def get_custom_rates(self):
        return self.custom_rates

    def get_rates_noisy(self):
        return self.custom_rates_noisy

int
def create_LIF_rate_neurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise):

    neurons=[]
    for i in range(n_neurons):
        n=LIFneuron(x_intercept_array[i],max_rate_array[i],encoders[i],tau_ref,tau_rc)
        n.set_sample_rates(x)
        n.set_sample_rates_noisy(x,noise)
        neurons.append(n)
    return neurons


def get_optimal_decoders_noisy(neurons,x,S,noise):

    # Use A=matrix of activities (the firing of each neuron for each x value)
    A_T=[]
    for n in neurons:
        A_T.append(n.get_sample_rates())
    A_T=np.matrix(A_T)
    A=np.transpose(A_T)
    x=np.matrix(x)
    upsilon=A_T*x/S
    gamma=A_T*A/S + np.identity(len(neurons))*noise**2
    d=np.linalg.inv(gamma)*upsilon
    return d


def get_state_estimate(neurons,x,d,noise):

    #check if the state to be estimated is equivalent to any of the stored firing
    #rate distributions held in the neuron class. If it is, there's no need to 
    #recompute the firing rates.
    xhat=[]
    if np.all(x == neurons[0].sample_x):
        for j in range(len(x)):
            xhat_i=0
            for i in range(len(neurons)):
                if noise != 0:
                    a_ij=neurons[i].get_sample_rates_noisy()[j]
                    d_i=np.array(d[i])
                    xhat_i+=(a_ij*d_i).flatten()
                else:
                    a_ij=neurons[i].get_sample_rates()[j]
                    d_i=np.array(d[i])
                    xhat_i+=(a_ij*d_i).flatten()
            xhat.append(xhat_i)
        xhat=np.array(xhat)

    elif np.all(x == neurons[0].custom_x):
        for j in range(len(x)):
            xhat_i=0
            for i in range(len(neurons)):
                if noise != 0:
                    a_ij=neurons[i].get_custom_rates()[j]
                    d_i=np.array(d[i])
                    xhat_i+=(a_ij*d_i).flatten()
                else:
                    a_ij=neurons[i].get_custom_rates()[j]
                    d_i=np.array(d[i])
                    xhat_i+=(a_ij*d_i).flatten()
            xhat.append(xhat_i)
        xhat=np.array(xhat)

    else:
        for n in neurons:
            n.set_custom_rates(x)
            n.set_custom_rates_noisy(x,noise)
            n.custom_x=x
        for j in range(len(x)):
            xhat_i=0
            for i in range(len(neurons)):
                if noise != 0:
                    a_ij=neurons[i].get_custom_rates()[j]
                    d_i=np.array(d[i])
                    xhat_i+=(a_ij*d_i).flatten()
                else:
                    a_ij=neurons[i].get_custom_rates()[j]
                    d_i=np.array(d[i])
                    xhat_i+=(a_ij*d_i).flatten()
            xhat.append(xhat_i)
        xhat=np.array(xhat)

    return xhat

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def one():

	#LIF parameters
    n_neurons=20
    min_fire_rate=100
    max_fire_rate=200
    min_x=-2
    max_x=2
    max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
    x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
    e=-1+2*np.random.randint(2,size=n_neurons)
    dx=0.05
    x=np.vstack(np.arange(min_x,max_x,dx))
    tau_ref=0.002
    tau_rc=0.02
    S=len(x)
    noise=0.1*np.max(max_rate_array)

    #create neurons
    neurons=create_LIF_rate_neurons(
    			n_neurons,x_intercept_array,max_rate_array,x,e,tau_ref,tau_rc,noise)

    #plot tuning curves
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for n in neurons:
        y=n.get_sample_rates_noisy()
        ax.plot(x,y)
    ax.set_xlim(min_x,max_x)
    ax.set_xlabel('x')
    ax.set_ylabel('Firing Rate $a$ (Hz)')
    plt.show()

    d=get_optimal_decoders_noisy(neurons,x,S,noise)
    xhat=get_state_estimate(neurons,x,d,noise)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x,x,'b',label='$x$')
    ax.plot(x,xhat,'g',label='$\hat{x}$')
    ax.set_ylim(-1,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\hat{x}$')
    legend=ax.legend(loc='best',shadow=True)
    plt.show()

def main():

	one()

main()
