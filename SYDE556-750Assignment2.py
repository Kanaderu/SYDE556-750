# Peter Duggins
# SYDE 556/750
# Jan 25, 2015
# Assignment 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

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



class spikingLIFneuron():

    def __init__(self,x1,x2,a1,a2,encoder,tau_ref,tau_rc):
        self.x1=x1
        self.x2=x2
        self.a1=a1
        self.a2=a2
        self.e=encoder
        self.tau_ref=tau_ref
        self.tau_rc=tau_rc
        self.Jbias=0
        self.alpha=0
        self.V=0
        self.dVdt=0
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

        self.sample_rates=[]
        self.sample_rates_noisy=[]
        self.custom_rates=[]
        self.custom_rates_noisy=[]
        self.custom_x=np.array([])

    def set_spikes(self,stimulus,T,dt):
        self.stimulus=stimulus #an array
        self.spikes=[]
        self.Vhistory=[]
        for t in range(len(stimulus)):
            self.dVdt=((1/self.tau_rc)*(self.Jbias-self.V)) + 1/dt*self.stimulus[t]  #differential equation for LIF dV/dt+ (norm)*stimulus(time=t)
            ref_window=int(self.tau_ref/dt)
            for h in range(ref_window):    #check if there have been spikes in the last t_rc seconds
                if len(self.spikes) >= ref_window and self.spikes[-h] == 1:
                    self.dVdt=0     #if so, voltage isn't allowed to change
            self.V=self.V+dt*self.dVdt  #Euler's Method Approximation V(t+1) = V(t) + dt *dV(t)/dt; the (norm) cancels the *dt for stim
            if self.V >= 1:
                self.spikes.append(1)   #a spike
                self.V=0    #reset
            else:
                self.spikes.append(0)   #not a spike
                if self.V < 0: self.V=0
            self.Vhistory.append(self.V)

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

def generate_signal(T,dt,rms,limit,seed):

    #first generate x_w, with the specified constraints, then use an inverse fft to get x_t
    rng=np.random.RandomState(seed=seed)
    t=np.arange(int(T/dt))*dt
    freq_vals = np.arange(int(T/dt))/T - (T/dt)/2 #in Hz
    w_vals = 2.0*np.pi*freq_vals #in radians
    x_w_half1=[]
    x_w_half2=[]

    for i in range(len(freq_vals)/2): #make half of X(w), those with negative freq
        if abs(freq_vals[i]) < limit:
            x_w_i_real = rng.normal(loc=0,scale=1)
            x_w_i_im = rng.normal(loc=0,scale=1)
        else:
            x_w_i_real = 0.0
            x_w_i_im = 0.0          
        x_w_half1.append(x_w_i_real + 1j*x_w_i_im)
        x_w_half2.append(x_w_i_real - 1j*x_w_i_im) #make the 2nd half of X(w) with complex conjugates 
   
    # print 'x_w half1',x_w_half1, '\nx_w half2',x_w_half2,
    x_w=np.concatenate((x_w_half1,x_w_half2[::-1]),axis=0) #reverse order to  preserve symmetry
    x_w=np.array(x_w)
    x_t=np.fft.ifft(np.fft.fftshift(x_w))
    true_rms=np.sqrt(1/T*np.sum(np.square(x_t)))
    x_t = x_t*rms/true_rms
    x_t = x_t.real  #ONLY return the real part, until I find the bug

    return x_t, x_w

def generate_smooth_signal(T,dt,rms,bandwidth,seed):

    rng=np.random.RandomState(seed=seed)
    t=np.arange(int(T/dt))*dt
    freq_vals = np.arange(int(T/dt))/T - (T/dt)/2 #in Hz
    w_vals = 2.0*np.pi*freq_vals #in radians
    x_w_half1=[]
    x_w_half2=[]

    for i in range(len(w_vals)/2): #make half of X(w), those with negative freq
        sigma=np.exp(-np.square(w_vals[i])/(2*np.square(bandwidth)))
        if sigma > np.finfo(float).eps: #distinguishable from zero
            x_w_i_real = rng.normal(loc=0,scale=sigma)
            x_w_i_im = rng.normal(loc=0,scale=sigma)       
        else:
            x_w_i_real = 0.0
            x_w_i_im = 0.0             
        x_w_half1.append(x_w_i_real + 1j*x_w_i_im)
        x_w_half2.append(x_w_i_real - 1j*x_w_i_im) #make the 2nd half of X(w) with complex conjugates 
   
    # print 'x_w half1',x_w_half1, '\nx_w half2',x_w_half2,
    # x_t_test=np.fft.irfft(np.fft.fftshift(np.array(x_w_half1)))
    # print 'x_t_test', x_t_test
    x_w=np.concatenate((x_w_half1,x_w_half2[::-1]),axis=0) #reverse order to  preserve symmetry
    x_w=np.array(x_w)
    x_t=np.fft.ifft(np.fft.fftshift(x_w))
    true_rms=np.sqrt(1/T*np.sum(np.square(x_t)))
    x_t = x_t*rms/true_rms
    x_t = x_t.real  #ONLY return the real part, until I find the bug

    return x_t, x_w

# ################################################################################################

def one_pt_one():

    T=1
    dt=0.001
    rms=0.5
    limit=10
    seed=1
    t=np.arange(int(T/dt))*dt
    freq = np.arange(int(T/dt))/T - (T/dt)/2

    limits=[5,10,20]
    x_t_list=[]
    x_w_list=[]
    ps_list=[]
    for i in range(len(limits)):  
        seed=i
        limit=limits[i]
        x_ti, x_wi = generate_signal(T,dt,rms,limit,seed)
        x_t_list.append(x_ti)
        x_w_list.append(x_wi)
        ps_list.append(np.abs(x_wi)**2)

    # print 'x_t', x_t, x_t.shape, 'rms=', np.sqrt(1/T*np.sum(np.square(x_t)))
    # print 'x_w', x_w, x_w.shape
    # print 'power spectrum', ps, ps.shape
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(len(limits)):  
        ax.plot(t,x_t_list[i],label='limit=%s' %int(limits[i]))
    # ax.plot(t,x_t.real,'b-', t, x_t.imag,'r--')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('x(t)')
    legend=ax.legend(loc='best',shadow=True)
    # ax=fig.add_subplot(212)
    # ax.plot(freq,ps)
    # ax.set_xlabel('Hz?')
    # ax.set_ylabel('x($\omega$)')
    # ax.set_xlim(-limit*2, limit*2)
    # plt.show()

    # #testing the fft methods, going from signal to power spectrum
    # vals=np.random.normal(size=len(t))
    # signal=np.array(vals)
    # x_w2=np.fft.fftshift(np.fft.fft(signal))
    # freq=np.fft.fftfreq(signal.size,d=dt)
    # power=np.abs(x_w2)**2
    # print 'test_signal',signal.shape
    # print 'test_x_w', x_w2.shape
    # print 'frequencies?', freq, freq.shape
    # print 'power?', power, power.shape
    # fig=plt.figure()
    # ax=fig.add_subplot(211)
    # ax.plot(signal)
    # ax.set_ylabel('x(t)')
    # ax=fig.add_subplot(212)
    # ax.plot(freq,power)
    # ax.set_ylabel('x($\omega$)')
    # plt.show()

    T=1
    dt=0.001
    rms=0.5
    limit=10
    avgs=100
    t=np.arange(int(T/dt))*dt
    freq = np.arange(int(T/dt))/T - (T/dt)/2 #in Hz
    w_vals = 2.0*np.pi*freq #in radians    
    x_w_list=[]
    for i in range(avgs):
        seed=i
        x_ti, x_wi = generate_signal(T,dt,rms,limit,seed)
        x_w_list.append(np.abs(x_wi))
    x_w_avg=np.average(x_w_list,axis=0)
    # print 'x_w_-1', x_w_list[-1]
    # print 'x_w_-2', x_w_list[-2]
    # print 'x_w_avg', x_w_avg
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(w_vals,x_w_avg)
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$|X(\omega)|$')
    ax.set_xlim(-limit*2*2*np.pi, limit*2*2*np.pi)
    plt.show()

def one_pt_two():

    #part a
    T=1
    dt=0.001
    rms=0.5
    seed=1
    t=np.arange(int(T/dt))*dt
    freq = np.arange(int(T/dt))/T - (T/dt)/2

    bandwidths=[5,10,20]
    x_t_list=[]
    for i in range(len(bandwidths)):  
        seed=i
        bandwidth=bandwidths[i]
        x_ti, x_wi = generate_smooth_signal(T,dt,rms,bandwidth,seed)
        x_t_list.append(x_ti)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(len(bandwidths)):  
        ax.plot(t,x_t_list[i],label='bandwidth=%s' %int(bandwidths[i]))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('x(t)')
    legend=ax.legend(loc='best',shadow=True)

    #part b
    T=1
    dt=0.001
    rms=0.5
    bandwidth=10
    avgs=100
    t=np.arange(int(T/dt))*dt
    freq = np.arange(int(T/dt))/T - (T/dt)/2 #in Hz
    w_vals = 2.0*np.pi*freq #in radians    
    x_w_list=[]
    for i in range(avgs):
        seed=i
        x_ti, x_wi = generate_smooth_signal(T,dt,rms,bandwidth,seed)
        x_w_list.append(np.abs(x_wi))
    x_w_avg=np.average(x_w_list,axis=0)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(w_vals,x_w_avg)
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('$|X(\omega)|$')
    ax.set_xlim(-bandwidth*2*2*np.pi, bandwidth*2*2*np.pi)
    plt.show()

def two():

    x1=0
    x2=1
    a1=40
    a2=150
    encoder=1
    tau_ref=0.002
    tau_rc=0.02
    T=1.0
    dt=0.001

    #1.1
    n1=spikingLIFneuron(x1,x2,a1,a2,encoder,tau_ref,tau_rc)
    stimulus1 = np.linspace(0,0,T/dt)  #constant stimulus of zero in an array
    n1.set_spikes(stimulus1,T,dt)
    spikes1=n1.get_spikes()
    stimulus2 = np.linspace(1,1,T/dt)  #constant stimulus of one
    n1.set_spikes(stimulus2,T,dt)
    spikes2=n1.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    times=np.arange(0,T,dt)
    ax.plot(times,spikes1, label='%s spikes' %np.count_nonzero(spikes1))
    ax.plot(times,spikes2, label='%s spikes' %np.count_nonzero(spikes2))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Voltage')
    ax.set_xlim(0,T)
    ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

    #1.2
    #the number of spikes we expect at stimulus=0 is zero, because voltage begins at zero, does not increase due to
    #dV/dT in LIF equation, and has no external stimulus. when stimulus = 1, enough stimulus is added at every 
    #timestep to fire an action potential. However, the refractory period keeps this from happening every step.
    #there are two dts in the space of tau_ref, so a spike can only be fired once every 3 steps. this means the spike
    #count should be steps/3 = 100/3 = 333, which is what I get.

    #1.3
    T=1
    dt=0.001
    rms=0.5
    limit=30
    seed=3
    t=np.arange(int(T/dt))*dt
    freq = np.arange(int(T/dt))/T - (T/dt)/2
    x_t, x_w = generate_signal(T,dt,rms,limit,seed)
    stimulus3 = np.array(x_t) * 10
    n1.set_spikes(stimulus3,T,dt)
    spikes3=n1.get_spikes()

    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    times=np.arange(0,T,dt)
    ax.plot(t,x_t, label='$x(t)$')
    ax.plot(t,spikes3, label='%s spikes' %np.count_nonzero(spikes3))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('signal')
    # ax.set_xlim(0,T)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

    print n1.Vhistory

    #1.4
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(111)
    times=np.arange(0,T,dt)
    ax.plot(t,x_t, label='$x(t)$')
    ax.plot(t,spikes3, label='%s spikes' %np.count_nonzero(spikes3))bon
    ax.plot(t,n1.Vhistory, label='Voltage')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('signal')
    ax.set_xlim(0,0.2)
    # ax.set_ylim(0,2)
    legend=ax.legend(loc='best') 
    plt.show()

    #Bonus Question

def main():

    # one_pt_one()
    # one_pt_two()
    two()

main()