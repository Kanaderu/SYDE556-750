# Peter Duggins
# SYDE 556/750
# March 3, 2015
# Assignment 3

import numpy as np
import matplotlib.pyplot as plt

class LIF_rate_neuron():

    # def __init__(self,x_int,x_max,a_max,e,tau_ref,tau_rc):
    #     self.x_int=x_int
    #     self.x_max=np.dot(x_max,e) #fix negative sign
    #     self.a_max=a_max
    #     self.e=e
    #     self.tau_ref=tau_ref
    #     self.tau_rc=tau_rc

    #     self.alpha=1/(np.dot(self.x_max,self.e)-np.dot(self.x_int,self.e))*\
    #                 (-1+1/(1-np.exp((self.tau_ref-1/self.a_max)/self.tau_rc)))
    #     self.Jbias=1-self.alpha*np.dot(self.x_int,self.e)

    #     self.sample_rates_noisy=[]
    #     self.custom_rates=[]
    #     self.custom_rates_noisy=[]
    #     self.custom_x=np.array([])

    def __init__(self,e,alpha,Jbias,tau_ref,tau_rc):
        self.e=e
        self.alpha=alpha
        self.Jbias=Jbias
        self.tau_ref=tau_ref
        self.tau_rc=tau_rc
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

    def set_sample_rates_noisy(self,x_vals,noise,rng):
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
                eta=rng.normal(loc=0,scale=noise)
            self.sample_rates_noisy.append(float(rate+eta))
        return self.sample_rates_noisy

    def get_sample_rates(self):
        return self.sample_rates

    def get_sample_rates_noisy(self):
        return self.sample_rates_noisy


class LIF_spiking_neuron():

    def __init__(self,e,alpha,Jbias,tau_ref,tau_rc):
        self.e=e
        self.tau_ref=tau_ref
        self.tau_rc=tau_rc
        self.Jbias=Jbias
        self.alpha=alpha
        self.V=0.0
        self.dVdt=0.0
        self.stimulus=[]
        self.spikes=[]
        self.Vhistory=[]

    def set_spikes(self,stimulus,T,dt):
        self.stimulus=stimulus #an array
        self.spikes=[]
        self.Vhistory=[]
        ref_window=int(self.tau_ref/dt)

        for t in range(len(stimulus)):
            self.J=self.alpha*np.dot(self.stimulus[t],self.e) + self.Jbias
            self.dVdt=((1/self.tau_rc)*(self.J-self.V))
            for h in range(ref_window):    #check if there have been spikes in the last t_rc seconds
                if len(self.spikes) >= ref_window and self.spikes[-(h+1)] == 1:
                    self.dVdt=0     #if so, voltage isn't allowed to change
            self.V=self.V+dt*self.dVdt  #Euler's Method Approximation V(t+1) = V(t) + dt *dV(t)/dt
            if self.V >= 1.0:
                self.spikes.append(1)   #a spike
                self.V=0.0    #reset
            else:
                self.spikes.append(0)   #not a spike
                if self.V < 0.0: self.V=0.0
            self.Vhistory.append(self.V)

    def get_spikes(self):
        return self.spikes

def generate_signal(T,dt,rms,limit,seed,rng,distribution='uniform'):

    #first generate x_w, with the specified constraints, then use an inverse fft to get x_t
    t=np.arange(int(T/dt))*dt
    delta_w = 2*np.pi/T
    w_vals = np.arange(-len(t)/2,0,delta_w) #make half of X(w), those with negative freq
    w_limit=2*np.pi*limit
    bandwidth=2*np.pi*limit
    # bandwidth=limit
    x_w_half1=[]
    x_w_half2=[]

    for i in range(len(w_vals)):
        if distribution=='uniform':
            if abs(w_vals[i]) < w_limit:
                x_w_i_real = rng.normal(loc=0,scale=1)
                x_w_i_im = rng.normal(loc=0,scale=1)
                x_w_half1.append(x_w_i_real + 1j*x_w_i_im)
                x_w_half2.append(x_w_i_real - 1j*x_w_i_im)  

        elif distribution=='gaussian':          
            sigma=np.exp(-np.square(w_vals[i])/(2*np.square(bandwidth)))
            if sigma > np.finfo(float).eps: #distinguishable from zero
                x_w_i_real = rng.normal(loc=0,scale=sigma)
                x_w_i_im = rng.normal(loc=0,scale=sigma)  
                x_w_half1.append(x_w_i_real + 1j*x_w_i_im)
                x_w_half2.append(x_w_i_real - 1j*x_w_i_im) 

    x_w_pos=np.hstack((x_w_half2[::-1],np.zeros(len(t)/2-len(x_w_half2))))
    x_w_neg=np.hstack((np.zeros(len(t)/2-len(x_w_half1)),x_w_half1))
    x_w=np.hstack(([0+0j],x_w_pos,x_w_neg)) 
    x_t=np.fft.ifft(x_w)
    true_rms=np.sqrt(dt/T*np.sum(np.square(x_t)))   
    x_t = x_t*rms/true_rms
    x_w = x_w*rms/true_rms

    return x_t.real, x_w     


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


def get_rate_estimate(neurons,x,d,noise):

    #check if the state to be estimated is equivalent to any of the stored firing
    #rate distributions held in the neuron class. If it is, there's no need to 
    #recompute the firing rates.
    xhat=[]
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
    return xhat

def get_estimate_smoothed(spikes,h,d,rms):

    xhat=np.sum([d[i]*np.convolve(spikes[i],h,mode='full')
        [:len(spikes[i])] for i in range(len(d))],axis=0)
    true_rms=np.sqrt(1.0/len(spikes[0])*np.sum(np.square(xhat))) #normalize using rms
    xhat=(xhat*rms/true_rms).T
    return xhat

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def one():

	#LIF parameters
    n_neurons=20
    a_min=100
    a_max=200
    x_min=-2
    x_max=2
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    seed=3

    rng1=np.random.RandomState(seed=seed) #for neuron parameters
    rng2=np.random.RandomState(seed=seed) #for white noise signal
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array=-1+2*rng1.randint(2,size=n_neurons)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))
    S=len(x_sample)
    noise=0.1*np.max(a_max_array)

    #create neurons
    neurons=[]
    for i in range(n_neurons):
        # Assume e=1 for calculation of alpha and Jbias
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        n=LIF_rate_neuron(e_array[i],alpha,Jbias,tau_ref,tau_rc)
        n.set_sample_rates(x_sample)
        n.set_sample_rates_noisy(x_sample,noise,rng1)
        neurons.append(n)

    #plot tuning curves
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    for n in neurons:
        y=n.get_sample_rates_noisy()
        ax.plot(x_sample,y)
    ax.set_xlim(x_min,x_max)
    ax.set_xlabel('x')
    ax.set_ylabel('Firing Rate $a$ (Hz)')
    plt.show()

    #compute decoders and state estimate
    d=get_optimal_decoders_noisy(neurons,x_sample,S,noise)
    xhat=get_rate_estimate(neurons,x_sample,d,noise)

    #plot signal and estimate
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(x_sample,x_sample,label='$x$')
    ax.plot(x_sample,xhat,label='$\hat{x}$')
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((x_sample-xhat)**2)))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\hat{x}$')
    legend=ax.legend(loc='best',shadow=True)
    plt.show()

def two():

    #LIF parameters
    n_neurons=20
    a_min=100
    a_max=200
    x_min=-2
    x_max=2
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    seed=3

    rng1=np.random.RandomState(seed=seed) #for neuron parameters
    rng2=np.random.RandomState(seed=seed) #for white noise signal
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array=-1+2*rng1.randint(2,size=n_neurons)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))
    S=len(x_sample)
    noise=0.1*np.max(a_max_array)

    #create rate neurons, until we find one with a(x=0) between 20-50HZ
    neurons=[]
    chosen_one=None
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        n=LIF_rate_neuron(e_array[i],alpha,Jbias,tau_ref,tau_rc)
        n.set_sample_rates(x_sample)
        n.set_sample_rates_noisy(x_sample,noise,rng1)
        if 20 < n.get_sample_rates_noisy()[len(x_sample)/2] < 50:
            chosen_one=n
            break

    if chosen_one == None:
        print "Warning: no suitible neurons created"
    else:
        alpha_chosen=chosen_one.alpha
        Jbias_chosen=chosen_one.Jbias

    e1=1
    e2=-1
    T=1
    dt=0.001
    rms=1
    limit=5
    seed=3
    n=0

    #create spiking neurons with chosen parameters
    n1=LIF_spiking_neuron(e1,alpha_chosen,Jbias_chosen,tau_ref,tau_rc)
    n2=LIF_spiking_neuron(e2,alpha_chosen,Jbias_chosen,tau_ref,tau_rc)
    #create equivalent rate neurons for computing decoders
    nr1=LIF_rate_neuron(e1,alpha_chosen,Jbias_chosen,tau_ref,tau_rc)
    nr2=LIF_rate_neuron(e2,alpha_chosen,Jbias_chosen,tau_ref,tau_rc)
    nr1.set_sample_rates(x_sample)
    nr1.set_sample_rates_noisy(x_sample,noise,rng1)
    nr2.set_sample_rates(x_sample)
    nr2.set_sample_rates_noisy(x_sample,noise,rng1)

    #generate white noise signal
    t=np.arange(int(T/dt)+1)*dt
    x_t, x_w = generate_signal(T,dt,rms,limit,seed,rng2,'uniform')
    Nt = len(x_t)                
    stimulus = np.array(x_t)
    n1.set_spikes(stimulus,T,dt)
    n2.set_spikes(stimulus,T,dt)
    spikes1=n1.get_spikes()
    spikes2=n2.get_spikes()
    spikes=np.array([spikes1,spikes2])

    #set post-synaptic current temporal filter
    tau_synapse=0.005
    freq = np.arange(Nt)/T - Nt/(2.0*T)   
    omega = freq*2*np.pi            
    spikes=np.array([spikes1,spikes2])
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.linalg.norm(h)

    #calculate decoders using rate prodecure and filtered state estimate
    rate_neurons=[nr1,nr2]
    d=get_optimal_decoders_noisy(rate_neurons,x_sample,S,noise)
    # d=get_decoders_smoothed(spikes,h,x_t)
    xhat=get_estimate_smoothed(spikes,h,d,rms)

    #plot signal, spikes, and estimate
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,x_t, label='$x(t)$ = smoothed white noise')
    ax.plot(t,(spikes[0]-spikes[1]), color='k', label='spikes', alpha=0.2)
    ax.plot(t,xhat, label='$\hat{x}(t)$, $\\tau$ = %s' %tau_synapse)
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((x_t-xhat)**2)))
    ax.set_xlabel('time (s)')
    legend=ax.legend(loc='best') 
    plt.show()

    #plot filter
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,h)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$h(t)$')
    ax.set_xlim(0,0.02)
    plt.tight_layout()
    plt.show()


def main():

	# one()
    two()

main()
