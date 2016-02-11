# Peter Duggins
# SYDE 556/750
# Jan 25, 2015
# Assignment 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


class spikingLIFneuron():

    def __init__(self,x1,x2,a1,a2,encoder,tau_ref,tau_rc):
        self.x1=float(x1)
        self.x2=float(x2)
        self.a1=float(a1)
        self.a2=float(a2)
        self.e=encoder
        self.tau_ref=tau_ref
        self.tau_rc=tau_rc
        self.Jbias=0.0
        self.alpha=0.0
        self.V=0.0
        self.dVdt=0.0
        self.stimulus=[]
        self.spikes=[]
        self.Vhistory=[]

        if x1==0:
            self.Jbias=1/(1-np.exp((self.tau_ref - 1/self.a1)/self.tau_rc))
            self.alpha=(1/np.dot(self.x2,self.e)) * (1/(1-np.exp((self.tau_ref - 1/self.a2)/self.tau_rc)) - self.Jbias)
        elif x2==0:
            self.Jbias=1/(1-np.exp((self.tau_ref - 1/self.a2)/self.tau_rc))
            self.alpha=(1/np.dot(self.x1,self.e)) * (1/(1-np.exp((self.tau_ref - 1/self.a1)/self.tau_rc)) - self.Jbias)
        else:
            self.Jbias=(1/(1-self.x2/self.x1)) - 1/(1-np.exp((self.tau_ref - 1/self.a2)/self.tau_rc)) - \
             (self.x2/self.x1) * 1/(1-np.exp((self.tau_ref - 1/self.a1)/self.tau_rc))
            self.alpha=(1/(np.dot(self.x2,self.e))) * (1/(1-np.exp((self.tau_ref - 1/self.a2)/self.tau_rc)) - self.Jbias)

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
        print self.alpha, np.dot(self.stimulus[-1],self.e), self.J

    def get_spikes(self):
        return self.spikes


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

def generate_signal(T,dt,rms,limit,seed,distribution='uniform'):

    #first generate x_w, with the specified constraints, then use an inverse fft to get x_t
    rng=np.random.RandomState(seed=seed)
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
                x_w_half2.append(x_w_i_real - 1j*x_w_i_im) #make the 2nd half of X(w) with complex conjugates 

    x_w_pos=np.hstack((x_w_half2[::-1],np.zeros(len(t)/2-len(x_w_half2))))
    # print len(w_vals)-len(x_w_half1), np.zeros(len(w_vals)-len(x_w_half1))
    # print len(w_vals)-len(x_w_half2), np.zeros(len(w_vals)-len(x_w_half2))
    x_w_neg=np.hstack((np.zeros(len(t)/2-len(x_w_half1)),x_w_half1))
    # print 'x_w_positive',x_w_pos
    # print 'x_w_negative',x_w_neg
    x_w=np.hstack(([0+0j],x_w_pos,x_w_neg)) #amplitudes corresponding to [w_0, w_pos increasing, w_neg increasing]
    x_t=np.fft.ifft(x_w)
    true_rms=np.sqrt(dt/T*np.sum(np.square(x_t)))
    x_t = x_t*rms/true_rms

    # print 'default precision'
    # print 'signal after ifft', x_t
    # print 'signal`s imaginary values (signal.imag)', x_t.imag
    # print 'sum of imaginary components in signal (x_t.imag.sum())', x_t.imag.sum()
    # # np.set_printoptions(precision=10)
    # print 'precision=10'
    # print 'signal after ifft', x_t
    # print 'signal`s imaginary values (signal.imag)', x_t.imag
    # print 'sum of imaginary components in signal (x_t.imag.sum())', x_t.imag.sum()

    return x_t.real, x_w     #return real part of signal to avoid bug, but I prmose they are less than e-15

# ################################################################################################

def one_pt_one_a():

    T=1
    dt=0.001
    rms=0.5
    limit=10
    seed=1
    t=np.arange(int(T/dt)+1)*dt

    limits=[5,10,20]
    x_t_list=[]
    for i in range(len(limits)):  
        seed=i
        limit=limits[i]
        x_ti, x_wi = generate_signal(T,dt,rms,limit,seed,'uniform')
        x_t_list.append(x_ti)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(len(limits)):  
        ax.plot(t,x_t_list[i],label='limit=%sHz' %(limits[i]))
        # ax.plot(t,x_t_list[i],label='limit=%s*2$\pi$ radians' %(limits[i]))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$x(t)$')
    legend=ax.legend(loc='best',shadow=True)
    plt.show()

    # w_vals=np.fft.fftfreq(len(x_w_avg))*2*np.pi/dt
    # w_limit=2*np.pi*limit
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    # ax.plot(w_vals,np.abs(x_wi))
    # ax.set_xlabel('$\omega$')
    # ax.set_ylabel('x($\omega$)')
    # ax.set_xlim(-w_limit*2, w_limit*2)
    # plt.show()

def one_pt_one_b():

    T=1
    dt=0.001
    rms=0.5
    limit=10
    avgs=100

    x_w_list=[]
    for i in range(avgs):
        seed=i
        x_ti, x_wi = generate_signal(T,dt,rms,limit,seed,'uniform')
        x_w_list.append(np.abs(x_wi))
    x_w_avg=np.average(x_w_list,axis=0)
    w_vals=np.fft.fftfreq(len(x_w_avg))*2*np.pi/dt
    w_limit=2*np.pi*limit

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.sort(w_vals),np.fft.fftshift(x_w_avg))
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$|X(\omega)|$')
    ax.set_xlim(-w_limit*2, w_limit*2)
    plt.show()

def one_pt_two_a():

    T=1
    dt=0.001
    rms=0.5
    seed=1
    t=np.arange(int(T/dt)+1)*dt

    bandwidths=[5,10,20]
    x_t_list=[]
    for i in range(len(bandwidths)):  
        seed=i
        bandwidth=bandwidths[i]
        x_ti, x_wi = generate_signal(T,dt,rms,bandwidth,seed,'gaussian')
        x_t_list.append(x_ti)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(len(bandwidths)):  
        # ax.plot(t,x_t_list[i],label='bandwidths*2$\pi$ radians' %(bandwidths[i]))
        ax.plot(t,x_t_list[i],label='bandwidth=%s (Hz)' %(bandwidths[i]))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$x(t)$')
    legend=ax.legend(loc='best',shadow=True)
    plt.show()

def one_pt_two_b():

    T=1
    dt=0.001
    rms=0.5
    bandwidth=10
    avgs=100

    x_w_list=[]
    for i in range(avgs):
        seed=i
        x_ti, x_wi = generate_signal(T,dt,rms,bandwidth,seed,'gaussian')
        x_w_list.append(np.abs(x_wi))
    x_w_avg=np.average(x_w_list,axis=0)
    w_vals=np.fft.fftfreq(len(x_w_avg))*2*np.pi/dt
    w_limit=2*np.pi*bandwidth

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.sort(w_vals),np.fft.fftshift(x_w_avg))
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$|X(\omega)|$')
    ax.set_xlim(-w_limit*2, w_limit*2)
    plt.show()

def two_a():

    x1=0
    x2=1
    a1=40
    a2=150
    encoder=1
    tau_ref=0.002
    tau_rc=0.02
    T=1.0
    dt=0.001

    n1=spikingLIFneuron(x1,x2,a1,a2,encoder,tau_ref,tau_rc)
    stimulus1 = np.linspace(0.0,0.0,T/dt)  #constant stimulus of zero in an array
    n1.set_spikes(stimulus1,T,dt)
    spikes1=n1.get_spikes()
    stimulus2 = np.linspace(1.0,1.0,T/dt)  #constant stimulus of one
    n1.set_spikes(stimulus2,T,dt)
    spikes2=n1.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    times=np.arange(0,T,dt)
    ax.plot(times,spikes1, 'b', label='%s spikes' %np.count_nonzero(spikes1))
    ax.plot(times,spikes2, 'g', label='%s spikes' %np.count_nonzero(spikes2))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Voltage')
    # ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

    #2.2
    # The number of spikes we expect at stimulus=0 is 40Hz, because this is the firing rate corresponding to
    # x(t)=0 (no external stimulus) that we originally specified.
    # When stimulus = 1, we expect the firing rate corresponding to x(t)=1, which is 143Hz when we specified 
    # a rate of 150hz. Decreasing the timestep dt improves this accuracy (goes to 149 with dt=0.0001),
    # because Euler's method becomes more exact and because the refractory period is specified with more precision

def two_c():

    rms=0.5
    limit=30
    seed=3
    x1=0
    x2=1
    a1=40
    a2=150
    encoder=1
    tau_ref=0.002
    tau_rc=0.02
    T=1.0
    dt=0.001

    n1=spikingLIFneuron(x1,x2,a1,a2,encoder,tau_ref,tau_rc)
    t=np.arange(int(T/dt)+1)*dt
    x_t, x_w = generate_signal(T,dt,rms,limit,seed,'uniform')
    stimulus3 = np.array(x_t)
    n1.set_spikes(stimulus3,T,dt)
    spikes3=n1.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    times=np.arange(0,T,dt)
    ax.plot(t,x_t, label='$x(t)$')
    ax.plot(t,spikes3, label='%s spikes' %np.count_nonzero(spikes3))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('$x(t)$')
    # ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

def two_d():

    rms=0.5
    limit=30
    seed=3
    x1=0
    x2=1
    a1=40
    a2=150
    encoder=1
    tau_ref=0.002
    tau_rc=0.02
    T=1.0
    dt=0.001

    n1=spikingLIFneuron(x1,x2,a1,a2,encoder,tau_ref,tau_rc)
    t=np.arange(int(T/dt)+1)*dt
    x_t, x_w = generate_signal(T,dt,rms,limit,seed,'uniform')
    stimulus3 = np.array(x_t)
    n1.set_spikes(stimulus3,T,dt)
    spikes3=n1.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    times=np.arange(0,T,dt)
    ax.plot(t,x_t, label='$x(t)$')
    ax.plot(t,spikes3, label='spikes')
    ax.plot(t,n1.Vhistory, label='Voltage')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Value')
    ax.set_xlim(0,0.2)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

    #Bonus Question

def three_a():

    x1=0
    x2=1
    a1=40
    a2=150
    encoder1=1
    encoder2=-1
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    rms=0.5
    limit=30
    seed=3

    n1=spikingLIFneuron(x1,x2,a1,a2,encoder1,tau_ref,tau_rc)
    n2=spikingLIFneuron(x1,x2,a1,a2,encoder2,tau_ref,tau_rc)
    t=np.arange(int(T/dt)+1)*dt
    stimulus1 = np.linspace(0,0,T/dt+1)  #constant stimulus of zero in an array
    n1.set_spikes(stimulus1,T,dt)
    n2.set_spikes(stimulus1,T,dt)
    spikes1=n1.get_spikes()
    spikes2=n2.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,stimulus1, label='x(t)')
    ax.plot(t,spikes1, label='%s spikes' %np.count_nonzero(spikes1))
    ax.plot(t,spikes2, label='%s spikes' %np.count_nonzero(spikes2))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Voltage')
    ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

def three_b():

    x1=0
    x2=1
    a1=40
    a2=150
    encoder1=1
    encoder2=-1
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    rms=0.5
    limit=30
    seed=3

    n1=spikingLIFneuron(x1,x2,a1,a2,encoder1,tau_ref,tau_rc)
    n2=spikingLIFneuron(x1,x2,a1,a2,encoder2,tau_ref,tau_rc)
    t=np.arange(int(T/dt)+1)*dt
    stimulus2 = np.linspace(1,1,T/dt+1)
    n1.set_spikes(stimulus2,T,dt)
    n2.set_spikes(stimulus2,T,dt)
    spikes1=n1.get_spikes()
    spikes2=n2.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,stimulus2, label='x(t)')
    ax.plot(t,spikes1, label='%s spikes' %np.count_nonzero(spikes1))
    ax.plot(t,spikes2, label='%s spikes' %np.count_nonzero(spikes2))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Voltage')
    ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

def three_c():

    x1=0
    x2=1
    a1=40
    a2=150
    encoder1=1
    encoder2=-1
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    rms=0.5
    limit=30
    seed=3

    n1=spikingLIFneuron(x1,x2,a1,a2,encoder1,tau_ref,tau_rc)
    n2=spikingLIFneuron(x1,x2,a1,a2,encoder2,tau_ref,tau_rc)
    t=np.arange(int(T/dt)+1)*dt
    stimulus3 = 0.5*np.sin(10*np.pi*t)
    n1.set_spikes(stimulus3,T,dt)
    n2.set_spikes(stimulus3,T,dt)
    spikes1=n1.get_spikes()
    spikes2=n2.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,stimulus3, label='x(t)')
    ax.plot(t,spikes1, label='%s spikes' %np.count_nonzero(spikes1))
    ax.plot(t,spikes2, label='%s spikes' %np.count_nonzero(spikes2))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Voltage')
    ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

def three_d():

    x1=0
    x2=1
    a1=40
    a2=150
    encoder1=1
    encoder2=-1
    tau_ref=0.002
    tau_rc=0.02
    T=1
    dt=0.001
    rms=0.5
    limit=30
    seed=3
    T=1.0
    dt=0.001


    n1=spikingLIFneuron(x1,x2,a1,a2,encoder1,tau_ref,tau_rc)
    n2=spikingLIFneuron(x1,x2,a1,a2,encoder2,tau_ref,tau_rc)
    t=np.arange(int(T/dt)+1)*dt
    x_t, x_w = generate_signal(T,dt,rms,limit,seed,'uniform')
    stimulus4 = np.array(x_t)
    n1.set_spikes(stimulus4,T,dt)
    n2.set_spikes(stimulus4,T,dt)
    spikes1=n1.get_spikes()
    spikes2=n2.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    ax.plot(t,stimulus4, label='x(t)')
    ax.plot(t,spikes1, label='%s spikes' %np.count_nonzero(spikes1))
    ax.plot(t,spikes2, label='%s spikes' %np.count_nonzero(spikes2))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Voltage')
    ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

def main():

    # one_pt_one_a()
    # one_pt_one_b()
    # one_pt_two_a()
    # one_pt_two_b()
    # two_a()
    # two_c()
    # two_d()
    # three_a()
    three_b()
    # three_c()
    # three_d()

main()