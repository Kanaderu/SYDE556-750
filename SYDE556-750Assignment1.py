# Peter Duggins
# SYDE 556/750
# Jan 25, 2015
# Assignment 1

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

class ReLUneuron():

    def __init__(self,x_intercept,max_firing_rate,encoder):
        self.xintercept=x_intercept
        self.maxrate=max_firing_rate
        self.e=encoder
        self.alpha=self.maxrate/(1 - self.xintercept)    #alpha=slope=(y2-y1)/(x2-x1)
        self.Jbias=self.maxrate-self.alpha
        self.sample_rates=[]
        self.sample_rates_noisy=[]
        self.custom_rates=[]
        self.custom_rates_noisy=[]
        self.custom_x=np.array([])

    def set_sample_rates(self,x_vals):
        self.sample_x=x_vals
        self.sample_rates=[]
        for x in x_vals:
            J=np.maximum(0,self.alpha*np.dot(x,self.e)+self.Jbias)
            self.sample_rates.append(float(np.maximum(0,J)))
        return self.sample_rates

    def set_sample_rates_noisy(self,x_vals,noise):
        self.sample_x=x_vals
        self.sample_rates_noisy=[]
        for x in x_vals:
            eta=0
            J=np.maximum(0,self.alpha*np.dot(x,self.e)+self.Jbias)
            if noise !=0:
                eta=np.random.normal(loc=0,scale=noise)
            self.sample_rates_noisy.append(float(np.maximum(0,J)+eta))
        return self.sample_rates_noisy

    def get_sample_rates(self):
        return self.sample_rates

    def get_sample_rates_noisy(self):
        return self.sample_rates_noisy

    def set_custom_rates(self,x_vals):
        self.custom_x=x_vals
        self.custom_rates=[]
        for x in x_vals:
            J=np.maximum(0,self.alpha*np.dot(x,self.e)+self.Jbias)
            self.custom_rates.append(float(np.maximum(0,J)))
        return self.custom_rates

    def set_custom_rates_noisy(self,x_vals,noise):
        self.custom_x=x_vals
        self.custom_rates_noisy=[]
        for x in x_vals:
            eta=0
            J=np.maximum(0,self.alpha*np.dot(x,self.e)+self.Jbias)
            if noise !=0:
                eta=np.random.normal(loc=0,scale=noise)
            self.custom_rates_noisy.append(float(np.maximum(0,J)+eta))
        return self.custom_rates_noisy

    def get_custom_rates(self):
        return self.custom_rates

    def get_custom_rates_noisy(self):
        return self.custom_rates_noisy

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

def ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise):

    neurons=[]
    for i in range(n_neurons):
        n=ReLUneuron(x_intercept_array[i],max_rate_array[i],encoders[i])
        n.set_sample_rates(x)
        n.set_sample_rates_noisy(x,noise)
        neurons.append(n)
    return neurons

def LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise):

    neurons=[]
    for i in range(n_neurons):
        n=LIFneuron(x_intercept_array[i],max_rate_array[i],encoders[i],tau_ref,tau_rc)
        n.set_sample_rates(x)
        n.set_sample_rates_noisy(x,noise)
        neurons.append(n)
    return neurons

def neuron_responses(neurons,x,noise):

    fig=plt.figure()
    ax=fig.add_subplot(111)
    for n in neurons:
        if noise != 0:
            y=n.get_sample_rates_noisy()
        else:
            y=n.get_sample_rates()
        ax.plot(x,y)
    ax.set_xlim(-1,1)
    ax.set_xlabel('x')
    ax.set_ylabel('Firing Rate $a$ (Hz)')
    plt.show()

def get_optimal_decoders(neurons,x,S):

    A_T=[]
    for n in neurons:
        A_T.append(n.get_sample_rates())
    A_T=np.matrix(A_T)
    A=np.transpose(A_T)
    x=np.matrix(x)
    upsilon=A_T*x/S
    gamma=A_T*A/S
    d=np.linalg.pinv(gamma)*upsilon
    return d

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

def error_vs_neurons(N_list,min_fire_rate,max_fire_rate,min_x,max_x,x,noise_mag,averages,tau_ref,tau_rc,n_type):

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
            d=get_optimal_decoders_noisy(neurons,x,S,noise)    #noisy optimization with noisy rates
            E_dist_n.append(get_e_dist(neurons,x,d,0))    #no noise
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
    x=np.vstack(np.arange(min_x,max_x,dx))

    neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
    neuron_responses(neurons,x,noise)

    S=len(x)
    d=get_optimal_decoders(neurons,x,S)    #noiseless optimization with noiseless rates
    print 'decoders:', d
    xhat=get_state_estimate(neurons,x,d,noise)    #noiseless rates

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

def one_pt_one_d():    #1.1d

    n_neurons=16
    min_fire_rate=100
    max_fire_rate=200
    min_x=-1
    max_x=1
    max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
    x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
    encoders=-1+2*np.random.randint(2,size=n_neurons)
    dx=0.05
    x=np.vstack(np.arange(min_x,max_x,dx))
    S=len(x)

    noise=0.2*np.max(max_rate_array)
    neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
    neuron_responses(neurons,x,noise)
    d=get_optimal_decoders(neurons,x,S)    #noiseless optimization with noisy rates
    # print d
    xhat=get_state_estimate(neurons,x,d,noise)    #noisy rates

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

def one_pt_one_e():    #1.1e
    
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
    x=np.vstack(np.arange(min_x,max_x,dx))
    S=len(x)

    noise=0.2*np.max(max_rate_array)
    neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)
    neuron_responses(neurons,x,noise)
    d1=get_optimal_decoders(neurons,x,S)        #noiseless optimization with noisy rates
    d2=get_optimal_decoders_noisy(neurons,x,S,noise)    #noisy optimization with noisy rates
    xhat1=get_state_estimate(neurons,x,d1,noise)    #noisy rates
    xhat2=get_state_estimate(neurons,x,d2,noise)    #noisy rates

    fig=plt.figure()
    ax=fig.add_subplot(211)
    ax.plot(x,x,'b',label='$x$')
    ax.plot(x,xhat1,'g',label='$\hat{x}$ d w/o noise')
    ax.set_ylim(-1,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\hat{x}$')
    legend=ax.legend(loc='best',shadow=True)
    ax=fig.add_subplot(212)
    ax.plot(x,x-xhat1)
    ax.set_xlim(-1,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$x - \hat{x}$')
    legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat1)**2))],loc='best')

    fig=plt.figure()
    ax=fig.add_subplot(211)
    ax.plot(x,x,'b',label='$x$')
    ax.plot(x,xhat2,'g',label='$\hat{x}$ d w/ noise')
    ax.set_ylim(-1,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\hat{x}$')
    legend=ax.legend(loc='best',shadow=True)
    ax=fig.add_subplot(212)
    ax.plot(x,x-xhat2)
    ax.set_xlim(-1,1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$x - \hat{x}$')
    legend=ax.legend(['RMSE=%f' %np.sqrt(np.average((x-xhat2)**2))],loc='best') 
    plt.show()

def one_pt_one_f():     #1.1f

    n_neurons=16
    min_fire_rate=100
    max_fire_rate=200
    min_x=-1
    max_x=1
    max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
    x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
    encoders=-1+2*np.random.randint(2,size=n_neurons)
    dx=0.05
    x=np.vstack(np.arange(min_x,max_x,dx))
    S=len(x)
    #noisy activity is calculated when neurons are created, so I do that here, then call
    #either get_rates() or get_rates_noisy as appropriate
    noise=0.2*np.max(max_rate_array)
    neurons=ReLUneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,noise)

    #noiseless activity, noiseless decoding
    d1=get_optimal_decoders(neurons,x,S)
    xhat1=get_state_estimate(neurons,x,d1,0)
    rmse1=np.sqrt(np.average((x-xhat1)**2))
    #noiseless activity, noisy decoding
    d2=get_optimal_decoders_noisy(neurons,x,S,noise)
    xhat2=get_state_estimate(neurons,x,d2,0)
    rmse2=np.sqrt(np.average((x-xhat2)**2))
    #noisy activity, noiseless decoding
    d3=get_optimal_decoders(neurons,x,S)
    xhat3=get_state_estimate(neurons,x,d3,noise)
    rmse3=np.sqrt(np.average((x-xhat3)**2))
    #noisy activity, noisy decoding
    d4=get_optimal_decoders_noisy(neurons,x,S,noise)
    xhat4=get_state_estimate(neurons,x,d4,noise)
    rmse4=np.sqrt(np.average((x-xhat4)**2))

    rmse_table_code=np.matrix([
        ['clean a, clean d',
         'clean a, noisy d'],
        ['noisy a, clean d',
         'noisy a, noisy d']
        ])
    rmse_table=np.matrix([[rmse1, rmse2], [rmse3, rmse4]])
    print rmse_table_code
    print rmse_table

def one_pt_two(): #1.2a-b

    N_list=[4,8,16,32,64,128,256,512]
    averages=20
    dx=0.05
    min_fire_rate=100
    max_fire_rate=200
    min_x=-0.95
    max_x=0.95
    tau_ref=0.002
    tau_rc=0.02
    x=np.vstack(np.arange(min_x,max_x,dx))

    noise_mag=0.1
    E_dist1,E_noise1 = error_vs_neurons(N_list,min_fire_rate,max_fire_rate,min_x,max_x,x,noise_mag,averages,tau_ref,tau_rc,'ReLU')
    noise_mag=0.01
    E_dist2,E_noise2 = error_vs_neurons(N_list,min_fire_rate,max_fire_rate,min_x,max_x,x,noise_mag,averages,tau_ref,tau_rc,'ReLU')

    fig=plt.figure()
    ax=fig.add_subplot(211)
    ax.loglog(N_list,E_dist1,'b',label='$E_{dist}$, $\\sigma=0.1$')
    ax.loglog(N_list,E_dist2,'r',label='$E_{dist}$, $\\sigma=0.01$')
    ax.loglog(N_list,1./np.square(N_list),'g',label='$1/n^2$')
    ax.set_xlabel('neurons')
    ax.set_ylabel('$E_{dist}$')
    legend=ax.legend(loc='best',shadow=True)
    ax=fig.add_subplot(212)
    ax.loglog(N_list,E_noise1,'b',label='$E_{noise}$, $\\sigma=0.1$')
    ax.loglog(N_list,E_noise2,'r',label='$E_{noise}$, $\\sigma=0.01$')
    ax.loglog(N_list,1./np.array(N_list),'g',label='$1/n$')
    legend=ax.legend(loc='best',shadow=True)
    ax.set_xlabel('neurons')
    ax.set_ylabel('$E_{noise}$')
    plt.show()

def one_pt_three():    #1.3

    n_neurons=16
    min_fire_rate=100
    max_fire_rate=200
    min_x=-1
    max_x=1
    max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
    x_intercept_array=np.random.uniform(min_x,max_x,n_neurons)
    encoders=-1+2*np.random.randint(2,size=n_neurons)
    dx=0.05
    x=np.vstack(np.arange(min_x,max_x,dx))
    tau_ref=0.002
    tau_rc=0.02
    S=len(x)
    noise=0.0
    neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
    neuron_responses(neurons,x,noise)

    noise=0.2*np.max(max_rate_array)
    neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
    neuron_responses(neurons,x,noise)
    d1=get_optimal_decoders(neurons,x,S)        #noiseless optimization with noisy rates
    d2=get_optimal_decoders_noisy(neurons,x,S,noise)    #noisy optimization with noisy rates
    xhat1=get_state_estimate(neurons,x,d1,noise)    #noisy rates
    xhat2=get_state_estimate(neurons,x,d2,noise)    #noisy rates

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

def two_pt_one():

    n_neurons=1
    max_fire_rate=100
    max_rate_array=[max_fire_rate]
    x1_min=-1
    x1_max=1
    x2_min=-1
    x2_max=1
    dx=0.05
    x_intercept_array=[[0,0]]
    encoders=[[1,-1]]    #preferred direction theta=-pi/4
    encoders=encoders/np.linalg.norm(encoders)
    # print encoders
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    # S=len(x)
    noise=0.0

    x1 = np.arange(x1_min, x1_max, dx)
    x2 = np.arange(x2_min, x2_max, dx)
    x=np.vstack(np.meshgrid(x1, x2)).reshape(2,-1).T
    neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Firing Rate (Hz)')
    for i in range(len(x1)):
        xs=x1[i]
        for j in range(len(x2)):
            ys=x2[j]
            zs = neurons[0].get_sample_rates()[i*len(x2)+j]
            ax.scatter(xs,ys,zs)
    plt.show()

    angles=np.linspace(0,2*np.pi,100)
    x=[[np.cos(a),np.sin(a)] for a in angles]

    neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
    y=neurons[0].get_sample_rates()
    def fit_cos(theta, A,B,C,D):
        return A*np.cos(B*theta+C)+D     #fitted cosine
    def fit_rec_cos(theta, A,B,C,D):
        return np.maximum(A*np.cos(B*theta+C)+D,0)     #fitted rectified cosine
    popt,pconv = curve_fit(fit_cos,angles,y)
    popt_rec,pconv_rec = curve_fit(fit_rec_cos,angles,y)
    y_fit=fit_cos(angles,*popt)
    y_fit_rec=fit_rec_cos(angles,*popt_rec)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\\theta$ (radians)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.plot(angles/(2*np.pi),y,label='Neural Responses')
    ax.plot(angles/(2*np.pi),y_fit,label='Fitted Cosine')
    ax.plot(angles/(2*np.pi),y_fit_rec,label='Fitted Rectified Cosine')
    legend=ax.legend(loc='best',shadow=True)
    plt.show()

def two_pt_two_a_thru_b():     #2.2a-b

    points=100
    n_neurons=100
    angles=np.random.uniform(0,2*np.pi,n_neurons)
    encoders=np.array([[np.cos(a),np.sin(a)] for a in angles])

    min_fire_rate=100
    max_fire_rate=200
    max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
    x1_min=-1
    x1_max=1
    x2_min=-1
    x2_max=1
    x_intercept_array=[[np.random.uniform(x1_min,x1_max),np.random.uniform(x2_min,x2_max)] for n in range(n_neurons)]
    dx=0.05
    x1 = np.arange(x1_min, x1_max, dx)
    x2 = np.arange(x2_min, x2_max, dx)
    x=np.vstack(np.meshgrid(x1, x2)).reshape(2,-1).T
    tau_ref=0.002
    tau_rc=0.02
    S=len(x)
    noise=0.2*np.max(max_rate_array)

    neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x,encoders,tau_ref,tau_rc,noise)
    d=get_optimal_decoders_noisy(neurons,x,S,noise)    #noisy optimization with noisy rates

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for e in encoders:
        ax.plot([0,e[0]],[0,e[1]])
    legend=ax.legend(['encoders'],loc='best',shadow=True)
    ax = fig.add_subplot(212)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for di in np.array(d):
        ax.plot([0,di[0]],[0,di[1]])
    legend=ax.legend(['decoders'],loc='best',shadow=True)
    plt.show()

def two_pt_two_c_thru_d():     #2.2c

    points=20
    n_neurons=100
    min_fire_rate=100
    max_fire_rate=200
    x1_min=-1
    x1_max=1
    x2_min=-1
    x2_max=1
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    x1 = np.arange(x1_min, x1_max, dx)
    x2 = np.arange(x2_min, x2_max, dx)
    x_sample=np.vstack(np.meshgrid(x1, x2)).reshape(2,-1).T
    max_rate_array=np.random.uniform(min_fire_rate,max_fire_rate,n_neurons)
    x_intercept_array=[[np.random.uniform(x1_min,x1_max),np.random.uniform(x2_min,x2_max)] for n in range(n_neurons)]
    noise=0.2*np.max(max_rate_array)

    angles=np.random.uniform(0,2*np.pi,points)     #note: does not distribute uniformally over the unit circle
    radii=np.random.uniform(0,1,points)     #but creates a high density of points near the center
    x=np.vstack([radii*np.cos(angles),radii*np.sin(angles)]).reshape(2,-1).T
    S=len(x)

    angles_enc=np.random.uniform(0,2*np.pi,n_neurons)     #note: does not distribute uniformally over the unit circle
    encoders=np.array([[np.cos(a),np.sin(a)] for a in angles_enc])
    S=len(x)

    neurons=LIFneurons(n_neurons,x_intercept_array,max_rate_array,x_sample,encoders,tau_ref,tau_rc,noise)
    d=get_optimal_decoders_noisy(neurons,x_sample,S,noise)    #noisy optimization with noisy rates
    xhat=get_state_estimate(neurons,x,d,noise)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    for xi in x:
        ax.plot([0,xi[0]],[0,xi[1]],'b')
    for xhati in xhat:
        ax.plot([0,xhati[0]],[0,xhati[1]],'g')
    legend=ax.legend(['$x$','$\hat{x}$, RMSE=%f' %np.sqrt(np.average((x-xhat)**2))],loc='best',shadow=True)
    plt.show()

    #2.2d
    xhat2=get_state_estimate(neurons,x,encoders,noise)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    for xi in xhat2:
        ax.plot([0,xi[0]],[0,xi[1]],'b')
    legend=ax.legend(['$\hat{x}$ w/ encoders, RMSE=%f' %np.sqrt(np.average((x-xhat2)**2))],loc='best',shadow=True)
    plt.show()

    
    xhat3=np.array([xi/np.linalg.norm(xi) for xi in xhat])        
    xhat4=np.array([xi/np.linalg.norm(xi) for xi in xhat2])
    
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    for xhat3i in xhat3:
        ax.plot([0,xhat3i[0]],[0,xhat3i[1]],'b')
    for xhat4i in xhat4:
        ax.plot([0,xhat4i[0]],[0,xhat4i[1]],'g')
    legend=ax.legend(
        ['$\hat{x}$ w/ decoders, normalized, RMSE=%f' %np.sqrt(np.average((x-xhat3)**2)),
        '$\hat{x}$ w/ encoders, normalized, RMSE=%f' %np.sqrt(np.average((x-xhat4)**2))],
        loc='best', shadow=True) 
    plt.show()
        
def main():

    one_pt_one_a_thru_c()
    one_pt_one_d()
    one_pt_one_e()
    one_pt_one_f()
    one_pt_two()
    one_pt_three()
    two_pt_one()
    two_pt_two_a_thru_b()
    two_pt_two_c_thru_d()

main()