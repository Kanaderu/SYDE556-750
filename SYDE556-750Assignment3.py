# Peter Duggins
# SYDE 556/750
# March 3, 2015
# Assignment 3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 24

class LIF_rate_neuron():

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
        ref_window=int(self.tau_ref/dt) #number of timesteps in refractory period

        for t in range(len(stimulus)):
            self.J=self.alpha*np.dot(self.stimulus[t],self.e) + self.Jbias
            self.dVdt=((1/self.tau_rc)*(self.J-self.V))
            #check if there have been spikes in the last tau_rc seconds
            for h in range(ref_window):   
                if len(self.spikes) >= ref_window and self.spikes[-(h+1)] == 1:
                    self.dVdt=0     #if so, voltage isn't allowed to change
            #Euler's Method Approximation V(t+1) = V(t) + dt *dV(t)/dt
            self.V=self.V+dt*self.dVdt  
            if self.V >= 1.0:
                self.spikes.append(1)   #a spike
                self.V=0.0    #reset
            else:
                self.spikes.append(0)   #not a spike
                if self.V < 0.0: self.V=0.0
            self.Vhistory.append(self.V)

    def get_spikes(self):
        return self.spikes

def ensemble(n_neurons,x_min,x_max,dx,a_min,a_max,
                seed,tau_ref,tau_rc,noise,T,dt,stimulus):

    #create sample points
    rng1=np.random.RandomState(seed=seed) #for neuron parameters
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array=-1+2*rng1.randint(2,size=n_neurons)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))

    #create neurons, rate neurons for decoders, spiking neurons for estimate
    rate_neurons=[]
    spiking_neurons=[]
    spikes=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(stimulus,T,dt)
        rate_neurons.append(nr)
        spiking_neurons.append(ns)
        spikes.append(ns.get_spikes())

    return rate_neurons, spiking_neurons, spikes

def generate_signal(T,dt,rms,limit,rng,distribution='uniform'):

    #first generate X(w), with the specified constraints, then use an inverse fft to get x_t
    # rng=np.random.RandomState(seed=seed)
    t=np.arange(int(T/dt))*dt
    delta_w = 2*np.pi/T #omega stepsize
    w_vals = np.arange(-len(t)/2,0,delta_w) #make half of X(w), those with negative freq
    w_limit=2*np.pi*limit #frequency in radians
    bandwidth=2*np.pi*limit #bandwidth in radians
    x_w_half1=[]
    x_w_half2=[]

    for i in range(len(w_vals)):    #loop over frequency values
        if distribution=='uniform':
            #if |w| is within the specified limit, generate a coefficient
            if abs(w_vals[i]) < w_limit:    
                x_w_i_real = rng.normal(loc=0,scale=1)  #mean=0, sigma=1
                x_w_i_im = rng.normal(loc=0,scale=1)
                x_w_half1.append(x_w_i_real + 1j*x_w_i_im)
                #make the 2nd half of X(w) with complex conjugates
                x_w_half2.append(x_w_i_real - 1j*x_w_i_im)  

        elif distribution=='gaussian':
            #draw sigma from a gaussian distribution
            sigma=np.exp(-np.square(w_vals[i])/(2*np.square(bandwidth)))
            if sigma > np.finfo(float).eps: #distinguishable from zero
                x_w_i_real = rng.normal(loc=0,scale=sigma)
                x_w_i_im = rng.normal(loc=0,scale=sigma)  
                x_w_half1.append(x_w_i_real + 1j*x_w_i_im)
                x_w_half2.append(x_w_i_real - 1j*x_w_i_im)  

    #zero pad the positive and negative amplitude lists, so each is len(samples/2)
    x_w_pos=np.hstack((x_w_half2[::-1],np.zeros(len(t)/2-len(x_w_half2))))
    x_w_neg=np.hstack((np.zeros(len(t)/2-len(x_w_half1)),x_w_half1))
    #assemble the symmetric X(w) according to numpy.fft documentation
    #amplitudes corresponding to [w_0, w_pos increasing, w_neg increasing]
    x_w=np.hstack(([0+0j],x_w_pos,x_w_neg))
    x_t=np.fft.ifft(x_w)
    #normalize time and frequency signals using RMS
    true_rms=np.sqrt(dt/T*np.sum(np.square(x_t)))   
    x_t = x_t*rms/true_rms
    #return real part of signal to avoid warning, but I promise they are less than e-15
    x_t = x_t.real.reshape(len(x_t),1)
    x_w = x_w*rms/true_rms

    return x_t, x_w  


def get_rate_decoders(neurons,x,noise):

    # Use A=matrix of activities (the firing of each neuron for each x value)
    A_T=[]
    for n in neurons:
        A_T.append(n.get_sample_rates())
    A_T=np.matrix(A_T)
    A=np.transpose(A_T)
    x=np.matrix(x)
    upsilon=A_T*x/len(x)
    gamma=A_T*A/len(x) + np.identity(len(neurons))*noise**2
    d=np.linalg.inv(gamma)*upsilon
    return d

def get_spike_decoders(spikes,h,x):

    A_T=np.array([np.convolve(s,h,mode='full')[:len(spikes[0])] for s in spikes]) #have to truncate from full here, same cuts off last half
    A=np.matrix(A_T).T
    x=np.matrix(x)
    upsilon=A_T*x/len(x)
    gamma=A_T*A/len(x)
    d=np.linalg.pinv(gamma)*upsilon
    return d

def get_function_decoders(neurons,x,noise,function):

    # Use A=matrix of activities (the firing of each neuron for each x value)
    A_T=[]
    for n in neurons:
        A_T.append(n.get_sample_rates())
    A_T=np.matrix(A_T)
    A=np.transpose(A_T)
    x=np.matrix(x)
    upsilon=A_T*function(x)/len(x)
    gamma=A_T*A/len(x) + np.identity(len(neurons))*noise**2
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

def get_spike_estimate(spikes,h,d):

    xhat=np.sum([d[i]*np.convolve(spikes[i],h,mode='full')
        [:len(spikes[i])] for i in range(len(d))],axis=0).T
    return xhat

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def one():

	#LIF parameters
    n_neurons=20
    a_min=100
    a_max=200
    x_min=-2
    x_max=2
    dx=0.05
    x_sample=np.vstack(np.arange(x_min,x_max,dx))
    tau_ref=0.002
    tau_rc=0.02
    seed=3
    T=1.0
    dt=0.001
    t=np.arange(int(T/dt)+1)*dt
    stimulus=np.zeros(len(t))
    noise=0.1*a_max

    rate_neurons, spiking_neurons, spikes = ensemble(
            n_neurons,x_min,x_max,dx,a_min,a_max,seed,
            tau_ref,tau_rc,noise,T,dt,stimulus)

    #plot tuning curves
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    for n in rate_neurons:
        y=n.get_sample_rates_noisy()
        ax.plot(x_sample,y)
    ax.set_xlim(x_min,x_max)
    ax.set_xlabel('x')
    ax.set_ylabel('Firing Rate $a$ (Hz)')
    plt.show()

    #compute decoders and state estimate
    d=get_rate_decoders(rate_neurons,x_sample,noise)
    xhat=get_rate_estimate(rate_neurons,x_sample,d,noise)

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
    noise=0.1*a_max

    rng1=np.random.RandomState(seed=seed) #for neuron parameters
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
    rng2=np.random.RandomState(seed=seed) #for white noise signal
    t=np.arange(int(T/dt)+1)*dt
    x_t, x_w = generate_signal(T,dt,rms,limit,rng2,'uniform')
    stimulus = np.array(x_t)
    n1.set_spikes(stimulus,T,dt)
    n2.set_spikes(stimulus,T,dt)
    spikes1=n1.get_spikes()
    spikes2=n2.get_spikes()
    spikes=np.array([spikes1,spikes2])

    #set post-synaptic current temporal filter
    tau_synapse=0.005          
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.sum(h*dt)  #normalize, effectively scaling spikes by dt

    #calculate decoders using rate prodecure and filtered state estimate
    rate_neurons=[nr1,nr2]
    d1=get_rate_decoders(rate_neurons,x_sample,S,noise)
    # d2=get_spike_decoders(spikes,h,x_t)
    xhat=get_spike_estimate(spikes,h,d1)

    #plot signal, spikes, and estimate
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,x_t, label='$x(t)$ (smoothed white noise)')
    # ax.plot(t,(spikes[0]-spikes[1]), color='k', label='spikes', alpha=0.2)
    ax.plot(t,xhat, label='$\hat{x}(t)$, $\\tau$ = %s' %tau_synapse)
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((x_t-xhat)**2)))
    ax.set_xlabel('time (s)')
    legend=ax.legend(loc='best') 
    plt.show()

    #plot filter
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,h, label='$h(t)$')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$h(t)$')
    ax.set_xlim(0,0.02)
    legend=ax.legend(loc='best') 
    plt.show()

def three():

    #parameters
    a_min=100
    a_max=200
    x_min=-2
    x_max=2
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    rms=1
    limit=5
    n=0
    avg=10
    t=np.arange(int(T/dt)+1)*dt

    #set post-synaptic current temporal filter
    tau_synapse=0.005          
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.sum(h*dt)  #normalize, effectively scaling spikes by dt

    n_neuron_list=[1,2,4,8,16,32,64,128,256]
    RMSE_list=[]

    for n in range(len(n_neuron_list)):
        n_neurons=n_neuron_list[n]
        RMSE_n=[]
        
        for a in range(avg):
            seed_neuron=n*avg+a
            seed_signal=n*3*a

            #generate white noise signal
            rng2=np.random.RandomState(seed=seed_signal) #for white noise signal
            x_t, x_w = generate_signal(T,dt,rms,limit,rng2,'uniform')

            #create sample points
            rng1=np.random.RandomState(seed=seed_neuron) #for neuron parameters
            a_max_array=rng1.uniform(a_min,a_max,n_neurons)
            x_int_array=rng1.uniform(x_min,x_max,n_neurons)
            e_array=-1+2*rng1.randint(2,size=n_neurons)
            x_sample=np.vstack(np.arange(x_min,x_max,dx))
            S=len(x_sample)
            noise=0.1*np.max(a_max_array)

            #create neurons, rate neurons for decoders, spiking neurons for estimate
            rate_neurons=[]
            spiking_neurons=[]
            spikes=[]
            for i in range(n_neurons):
                alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                        (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
                Jbias=1-alpha*np.dot(x_int_array[i],1)
                nr=LIF_rate_neuron(e_array[i],alpha,Jbias,tau_ref,tau_rc)
                ns=LIF_spiking_neuron(e_array[i],alpha,Jbias,tau_ref,tau_rc)
                nr.set_sample_rates(x_sample)
                nr.set_sample_rates_noisy(x_sample,noise,rng1)
                ns.set_spikes(x_t,T,dt)
                rate_neurons.append(nr)
                spiking_neurons.append(ns)
                spikes.append(ns.get_spikes())

            #calculate decoders using rate prodecure and filtered state estimate
            d=get_rate_decoders(rate_neurons,x_sample,S,noise)
            xhat=get_spike_estimate(spikes,h,d)

            #calculate RMSE
            RMSE_n_a=np.sqrt(np.average((x_t-xhat)**2))
            RMSE_n.append(RMSE_n_a)

        # plot signal, spikes, and estimate
        fig=plt.figure(figsize=(16,8))
        ax=fig.add_subplot(111)
        ax.plot(t,x_t, label='$x(t)$')
        ax.plot(t,xhat, label='$\hat{x}(t)$')
        ax.plot([],label='average RMSE=%f' %np.average(RMSE_n))
        ax.set_xlabel('time (s)')
        legend=ax.legend(loc='best') 
        plt.show()

        RMSE_list.append(np.average(RMSE_n))

    #plot RMSE vs n_neurons
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.loglog(n_neuron_list,RMSE_list, label='RMSE, $n_{avg}=%s$' %avg)
    ax.loglog(n_neuron_list,1/np.sqrt(n_neuron_list), label='$1/\sqrt{n}$')
    ax.set_xlabel('neurons')
    ax.set_ylabel('RMSE')
    legend=ax.legend(loc='best') 
    plt.show()

def four_a():

    #parameters
    n_neurons=200
    a_min=100
    a_max=200
    x_min=-1
    x_max=1
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    limit=5
    n=0
    seed_neuron=3
    seed_signal=3
    rng1=np.random.RandomState(seed=seed_neuron)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))  #firing rates sampled over all x
    S=len(x_sample)

    #generate linear ramp signal x(t)=t-1
    t=np.arange(int(T/dt)+1)*dt
    x_t=t-1

    #set post-synaptic current temporal filter
    tau_synapse=0.005          
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.sum(h*dt)  #normalize, effectively scaling spikes by dt

    #create first ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_1=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_1=[]
    spiking_neurons_1=[]
    spikes_1=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample) #firing rates sampled over all x
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(x_t,T,dt)  #find spikes for the signal x_t
        rate_neurons_1.append(nr)
        spiking_neurons_1.append(ns)
        spikes_1.append(ns.get_spikes())

    #calculate decoders for f(x)=2x+1, using x_sample, not x_t, to define rates
    function_1=lambda x: 2*x + 1
    d_1=get_function_decoders(rate_neurons_1,x_sample,S,noise,function_1)
    # d_11=get_rate_decoders(rate_neurons_1,x_sample,S,noise)
    f_xhat_1=get_spike_estimate(spikes_1,h,d_1)
    # f_xhat_11=get_spike_estimate(spikes_1,h,d_11)

    #create second ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_2=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_2=[]
    spiking_neurons_2=[]
    spikes_2=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(f_xhat_1,T,dt)  #find spikes for signal = previous decoded f(x)
        rate_neurons_2.append(nr)
        spiking_neurons_2.append(ns)
        spikes_2.append(ns.get_spikes())

    #calculate decoders for f(y)=y
    function_2=lambda y: y
    d_2=get_function_decoders(rate_neurons_2,x_sample,S,noise,function_2)
    yhat=get_spike_estimate(spikes_2,h,d_2)

    #plot signal, transformed signal, and estimate
    y_t=function_1(x_t)
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,x_t, label='$x(t)=t-1$')
    # ax.plot(t,f_xhat_1, label='$f(\hat{x}(t))_{transformed}$')
    # ax.plot(t,f_xhat_11, label='$\hat{x}(t)_{rate}$')
    ax.plot(t,y_t, label='$y(t)=2x(t)+1$')
    ax.plot(t,yhat, label='$\hat{y}(t)$')
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((y_t-yhat)**2)))
    ax.set_xlabel('time (s)')
    legend=ax.legend(loc='best') 
    plt.show()


def four_b():

    #parameters
    n_neurons=200
    a_min=100
    a_max=200
    x_min=-1
    x_max=1
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    limit=5
    n=0
    seed_neuron=3
    seed_signal=3
    rng1=np.random.RandomState(seed=seed_neuron)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))  #firing rates sampled over all x
    S=len(x_sample)

    #generate a randomly varying step signal
    rng2=np.random.RandomState(seed=seed_signal)
    t=np.arange(int(T/dt)+1)*dt
    x_t=np.array([np.full((len(t)/10),rng2.rand()-1) for i in range(10)]).flatten()
    x_t=np.insert(x_t,len(x_t),0).reshape(len(t),1)
 
    #set post-synaptic current temporal filter
    tau_synapse=0.005          
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.sum(h*dt)  #normalize, effectively scaling spikes by dt

    #create first ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_1=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_1=[]
    spiking_neurons_1=[]
    spikes_1=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample) #firing rates sampled over all x
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(x_t,T,dt)  #find spikes for the signal x_t
        rate_neurons_1.append(nr)
        spiking_neurons_1.append(ns)
        spikes_1.append(ns.get_spikes())

    #calculate decoders for f(x)=2x+1, using x_sample, not x_t, to define rates
    function_1=lambda x: 2*x + 1
    d_1=get_function_decoders(rate_neurons_1,x_sample,S,noise,function_1)
    # d_11=get_rate_decoders(rate_neurons_1,x_sample,S,noise)
    f_xhat_1=get_spike_estimate(spikes_1,h,d_1)
    # f_xhat_11=get_spike_estimate(spikes_1,h,d_11)

    #create second ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_2=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_2=[]
    spiking_neurons_2=[]
    spikes_2=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(f_xhat_1,T,dt)  #find spikes for signal = previous decoded f(x)
        rate_neurons_2.append(nr)
        spiking_neurons_2.append(ns)
        spikes_2.append(ns.get_spikes())

    #calculate decoders for f(y)=y
    function_2=lambda y: y
    d_2=get_function_decoders(rate_neurons_2,x_sample,S,noise,function_2)
    yhat=get_spike_estimate(spikes_2,h,d_2)

    #plot signal, transformed signal, and estimate
    y_t=function_1(x_t)
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,x_t, label='$x(t)$ (random step input)')
    # ax.plot(t,f_xhat_1, label='$f(\hat{x}(t))_{transformed}$')
    # ax.plot(t,f_xhat_11, label='$\hat{x}(t)_{rate}$')
    ax.plot(t,y_t, label='$y(t)=2x(t)+1$')
    ax.plot(t,yhat, label='$\hat{y}(t)$')
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((y_t-yhat)**2)))
    ax.set_xlabel('time (s)')
    legend=ax.legend(loc='best') 
    plt.show()

def four_c():

    #parameters
    n_neurons=200
    a_min=100
    a_max=200
    x_min=-1
    x_max=1
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    limit=5
    n=0
    seed_neuron=3
    seed_signal=3
    rng1=np.random.RandomState(seed=seed_neuron)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))  #firing rates sampled over all x
    S=len(x_sample)

    #generate a sinusoid
    t=np.arange(int(T/dt)+1)*dt
    x_t=0.2*np.sin(6*np.pi*t)
 
    #set post-synaptic current temporal filter
    tau_synapse=0.005          
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.sum(h*dt)  #normalize, effectively scaling spikes by dt

    #create first ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_1=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_1=[]
    spiking_neurons_1=[]
    spikes_1=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample) #firing rates sampled over all x
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(x_t,T,dt)  #find spikes for the signal x_t
        rate_neurons_1.append(nr)
        spiking_neurons_1.append(ns)
        spikes_1.append(ns.get_spikes())

    #calculate decoders for f(x)=2x+1, using x_sample, not x_t, to define rates
    function_1=lambda x: 2*x + 1
    d_1=get_function_decoders(rate_neurons_1,x_sample,S,noise,function_1)
    # d_11=get_rate_decoders(rate_neurons_1,x_sample,S,noise)
    f_xhat_1=get_spike_estimate(spikes_1,h,d_1)
    # f_xhat_11=get_spike_estimate(spikes_1,h,d_11)

    #create second ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_2=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_2=[]
    spiking_neurons_2=[]
    spikes_2=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(f_xhat_1,T,dt)  #find spikes for signal = previous decoded f(x)
        rate_neurons_2.append(nr)
        spiking_neurons_2.append(ns)
        spikes_2.append(ns.get_spikes())

    #calculate decoders for f(y)=y
    function_2=lambda y: y
    d_2=get_function_decoders(rate_neurons_2,x_sample,S,noise,function_2)
    yhat=get_spike_estimate(spikes_2,h,d_2)

    #plot signal, transformed signal, and estimate
    y_t=function_1(x_t)
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,x_t, label='$x(t)$ (random step input)')
    # ax.plot(t,f_xhat_1, label='$f(\hat{x}(t))_{transformed}$')
    # ax.plot(t,f_xhat_11, label='$\hat{x}(t)_{rate}$')
    ax.plot(t,y_t, label='$y(t)=2x(t)+1$')
    ax.plot(t,yhat, label='$\hat{y}(t)$')
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((y_t-yhat)**2)))
    ax.set_xlabel('time (s)')
    legend=ax.legend(loc='best') 
    plt.show()

    #discussion

def five():

#parameters
    n_neurons=200
    a_min=100
    a_max=200
    x_min=-1
    x_max=1
    dx=0.05
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    limit=5
    n=0
    seed_neuron=3
    seed_signal=3
    rng1=np.random.RandomState(seed=seed_neuron)
    x_sample=np.vstack(np.arange(x_min,x_max,dx))  #firing rates sampled over all x
    S=len(x_sample)

    #generate the sinusoidal signals x(t) and y(t)
    t=np.arange(int(T/dt)+1)*dt
    x_t=np.cos(3*np.pi*t)
    y_t=0.5*np.sin(2*np.pi*t)
    z_t=0.5*x_t+2*y_t

    #set post-synaptic current temporal filter
    tau_synapse=0.005          
    h=t**n*np.exp(-t/tau_synapse)
    h=h/np.sum(h*dt)  #normalize, effectively scaling spikes by dt

    #create first ensemble (rate neurons for decoders, spiking neurons for estimate)
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_1=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_1=[]
    spiking_neurons_1=[]
    spikes_1=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_1[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(x_t,T,dt)  
        rate_neurons_1.append(nr)
        spiking_neurons_1.append(ns)
        spikes_1.append(ns.get_spikes())

    #calculate decoders
    function_1=lambda x: 0.5*x
    d_1=get_function_decoders(rate_neurons_1,x_sample,S,noise,function_1)
    f_xhat=get_spike_estimate(spikes_1,h,d_1)

    #create second ensemble
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_2=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_2=[]
    spiking_neurons_2=[]
    spikes_2=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_2[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(y_t,T,dt)
        rate_neurons_2.append(nr)
        spiking_neurons_2.append(ns)
        spikes_2.append(ns.get_spikes())

    #calculate decoders for f(y)=y
    function_2=lambda y: 2*y
    d_2=get_function_decoders(rate_neurons_2,x_sample,S,noise,function_2)
    f_yhat=get_spike_estimate(spikes_2,h,d_2)

    #create third ensemble
    a_max_array=rng1.uniform(a_min,a_max,n_neurons)
    x_int_array=rng1.uniform(x_min,x_max,n_neurons)
    e_array_3=-1+2*rng1.randint(2,size=n_neurons)
    noise=0.1*np.max(a_max_array)
    rate_neurons_3=[]
    spiking_neurons_3=[]
    spikes_3=[]
    for i in range(n_neurons):
        alpha=1/(np.dot(x_max,1)-np.dot(x_int_array[i],1))*\
                (-1+1/(1-np.exp((tau_ref-1/a_max_array[i])/tau_rc)))
        Jbias=1-alpha*np.dot(x_int_array[i],1)
        nr=LIF_rate_neuron(e_array_3[i],alpha,Jbias,tau_ref,tau_rc)
        ns=LIF_spiking_neuron(e_array_3[i],alpha,Jbias,tau_ref,tau_rc)
        nr.set_sample_rates(x_sample)
        nr.set_sample_rates_noisy(x_sample,noise,rng1)
        ns.set_spikes(f_xhat+f_yhat,T,dt)  #input decoded outputs f(xhat)+f(yhat)
        rate_neurons_3.append(nr)
        spiking_neurons_3.append(ns)
        spikes_3.append(ns.get_spikes())

    #calculate decoders for f(y)=y
    function_3=lambda z: z
    d_3=get_function_decoders(rate_neurons_3,x_sample,S,noise,function_3)
    f_zhat=get_spike_estimate(spikes_3,h,d_3)


    #plot signal, transformed signal, and estimate
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,x_t, label='$x(t)=0.5cos(3\pi t)$')
    ax.plot(t,y_t, label='$y(t)=2sin(2\pi t)$')
    ax.plot(t,z_t, label='$z(t)=0.5x+2y$')
    ax.plot(t,f_zhat, label='$\hat{z}(t)$')
    ax.plot([],label='RMSE=%f' %np.sqrt(np.average((z_t-f_zhat)**2)))
    ax.set_xlabel('time (s)')
    legend=ax.legend(loc='best') 
    plt.show()


def main():

    one()
    # two()
    # three()
    # four_a()
    # four_b()
    # four_c()
    # five()


main()
