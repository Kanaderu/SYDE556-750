# Peter Duggins
# SYDE 556/750
# March 14, 2015
# Assignment 4

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 20

import nengo
from nengo.utils.ensemble import tuning_curves
from nengo.dists import Uniform
from nengo.solvers import LstsqNoise

def one_a():

	#ensemble parameters
	N=100
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.1
	lif_model=nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref)

	#model definition
	model = nengo.Network(label='1D Ensemble of LIF Neurons')
	with model:
		ens_1d = nengo.Ensemble(N,dimensions,
								intercepts=Uniform(-1.0,1.0),
								max_rates=Uniform(100,200),
								neuron_type=lif_model)

		#generate the decoders
		connection = nengo.Connection(ens_1d,ens_1d,
								solver=LstsqNoise(noise=noise))

	#create the simulator
	sim = nengo.Simulator(model)

	#retrieve evaluation points, activities, and decoders
	eval_points, activities = tuning_curves(ens_1d,sim)
	decoders = sim.data[connection].decoders.T

	#calculate the state estimate
	xhat = np.dot(activities,decoders)

	#plot tuning curves
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(211)
	ax.plot(eval_points,activities)
	# ax.set_xlabel('$x$')
	ax.set_ylabel('Firing Rate $a$ (Hz)')

	#plot representational accuracy
	ax=fig.add_subplot(212)
	ax.plot(eval_points,eval_points)
	ax.plot(eval_points,xhat,
		label='RMSE=%f' %np.sqrt(np.average((eval_points-xhat)**2)))
	ax.set_xlabel('$x$')
	ax.set_ylabel('$\hat{x}$')
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def one_b():

	#ensemble parameters
	N=100
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.1
	averages=5
	seed=3

	#objects and lists
	radii=np.logspace(-2,2,20)
	RMSE_list=[]
	RMSE_stddev_list=[]

	for i in range(len(radii)):

		RMSE_list_i=[]
		for a in range(averages):

			seed=3+a+i*len(radii)
			rng1=np.random.RandomState(seed=seed)
			lif_model=nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref)

			#model definition
			model = nengo.Network(label='1D LIF Ensemble',seed=seed)
			with model:
				ens_1d = nengo.Ensemble(N,dimensions,
										intercepts=Uniform(-1.0,1.0),
										max_rates=Uniform(100,200),
										radius=radii[i],
										#encoders=encoders,
										neuron_type=lif_model)

				#generate the decoders
				connection = nengo.Connection(ens_1d,ens_1d,
										solver=LstsqNoise(noise=noise))

			#create the simulator
			sim = nengo.Simulator(model)

			#retrieve evaluation points, activities, and decoders
			eval_points, activities = tuning_curves(ens_1d,sim)
			activities_noisy = activities + rng1.normal(
										scale=noise*np.max(activities),
										size=activities.shape)
			decoders = sim.data[connection].decoders.T

			#calculate the state estimate
			xhat = np.dot(activities_noisy,decoders)

			#calculate RMSE
			RMSE_list_i.append(np.sqrt(np.average((eval_points-xhat)**2)))

		RMSE_list.append(np.average(RMSE_list_i))
		RMSE_stddev_list.append(np.std(RMSE_list_i))

	#plot RMSE vs radius
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(np.log10(radii),RMSE_list)
	ax.fill_between(np.log10(radii),
		np.subtract(RMSE_list,RMSE_stddev_list),np.add(RMSE_list,RMSE_stddev_list),
		color='lightgray')
	ax.set_xlabel('log(radius)')
	ax.set_ylabel('RMSE')
	plt.show()

def one_c():

	#ensemble parameters
	N=100
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.1
	averages=10
	seed=3

	#objects and lists
	tau_refs=np.logspace(-6,-2.333,10)
	# tau_refs=np.linspace(0.0002,0.0049,10)
	RMSE_list=[]
	RMSE_stddev_list=[]
	eval_points_list=[]
	activities_list=[]

	for i in range(len(tau_refs)):

		RMSE_list_i=[]
		for a in range(averages):

			seed=3+a+i*len(tau_refs) #unique seed for each iteration
			rng1=np.random.RandomState(seed=seed)
			lif_model=nengo.LIF(tau_rc=tau_rc,tau_ref=tau_refs[i])

			#model definition
			model = nengo.Network(label='1D LIF Ensemble',seed=seed)
			with model:
				ens_1d = nengo.Ensemble(N,dimensions,
										intercepts=Uniform(-1.0,1.0),
										max_rates=Uniform(100,200),
										neuron_type=lif_model)

				#generate the decoders
				connection = nengo.Connection(ens_1d,ens_1d,
										solver=LstsqNoise(noise=noise))

			#create the simulator
			sim = nengo.Simulator(model)

			#retrieve evaluation points, activities, and decoders
			eval_points, activities = tuning_curves(ens_1d,sim)
			eval_points_list.append(eval_points)
			activities_list.append(activities)
			activities_noisy = activities + rng1.normal(
										scale=noise*np.max(activities),
										size=activities.shape)
			decoders = sim.data[connection].decoders.T

			#calculate the state estimate
			xhat = np.dot(activities_noisy,decoders)

			#calculate RMSE
			RMSE_list_i.append(np.sqrt(np.average((eval_points-xhat)**2)))

		RMSE_list.append(np.average(RMSE_list_i))
		RMSE_stddev_list.append(np.std(RMSE_list_i))

	#plot RMSE vs radius
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(np.log10(tau_refs),RMSE_list)
	ax.fill_between(np.log10(tau_refs),
		np.subtract(RMSE_list,RMSE_stddev_list),np.add(RMSE_list,RMSE_stddev_list),
		color='lightgray')
	ax.set_xlabel('log($\\tau_{ref}$)')
	ax.set_ylabel('RMSE')
	plt.show()

	# plot tuning curves
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(211)
	ax.plot(eval_points_list[0],activities_list[0])
	ax.set_title('$\\tau_{ref}=%f$' %tau_refs[0])
	# ax.set_xlabel('$x$')
	ax.set_ylabel('Firing Rate $a$ (Hz)')
	ax=fig.add_subplot(212)
	ax.plot(eval_points_list[-1],activities_list[-1])
	ax.set_title('$\\tau_{ref}=%f$' %tau_refs[-1])
	ax.set_xlabel('$x$')
	ax.set_ylabel('Firing Rate $a$ (Hz)')
	plt.show()

def one_d():

	#ensemble parameters
	N=100
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.1
	averages=5
	seed=3

	#objects and lists
	tau_rcs=np.logspace(-3,2,10)
	RMSE_list=[]
	RMSE_stddev_list=[]
	eval_points_list=[]
	activities_list=[]

	for i in range(len(tau_rcs)):

		RMSE_list_i=[]
		for a in range(averages):

			seed=3+a+i*len(tau_rcs) #unique seed for each iteration
			rng1=np.random.RandomState(seed=seed)
			lif_model=nengo.LIF(tau_rc=tau_rcs[i],tau_ref=tau_ref)

			#model definition
			model = nengo.Network(label='1D LIF Ensemble',seed=seed)
			with model:
				ens_1d = nengo.Ensemble(N,dimensions,
										intercepts=Uniform(-1.0,1.0),
										max_rates=Uniform(100,200),
										neuron_type=lif_model)

				#generate the decoders
				connection = nengo.Connection(ens_1d,ens_1d,
										solver=LstsqNoise(noise=noise))

			#create the simulator
			sim = nengo.Simulator(model)

			#retrieve evaluation points, activities, and decoders
			eval_points, activities = tuning_curves(ens_1d,sim)
			eval_points_list.append(eval_points)
			activities_list.append(activities)
			activities_noisy = activities + rng1.normal(
										scale=noise*np.max(activities),
										size=activities.shape)
			decoders = sim.data[connection].decoders.T

			#calculate the state estimate
			xhat = np.dot(activities_noisy,decoders)

			#calculate RMSE
			RMSE_list_i.append(np.sqrt(np.average((eval_points-xhat)**2)))

		RMSE_list.append(np.average(RMSE_list_i))
		RMSE_stddev_list.append(np.std(RMSE_list_i))

	#plot RMSE vs radius
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(np.log10(tau_rcs),RMSE_list)
	ax.fill_between(np.log10(tau_rcs),
		np.subtract(RMSE_list,RMSE_stddev_list),np.add(RMSE_list,RMSE_stddev_list),
		color='lightgray')
	# ax.plot(tau_refs,RMSE_list)
	ax.set_xlabel('log($\\tau_{RC}$)')
	ax.set_ylabel('RMSE')
	plt.show()

	# plot tuning curves
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(211)
	ax.plot(eval_points_list[0],activities_list[0])
	ax.set_title('$\\tau_{ref}=%f$' %tau_rcs[0])
	# ax.set_xlabel('$x$')
	ax.set_ylabel('Firing Rate $a$ (Hz)')
	ax=fig.add_subplot(212)
	ax.plot(eval_points_list[-1],activities_list[-1])
	ax.set_title('$\\tau_{ref}=%f$' %tau_rcs[-1])
	ax.set_xlabel('$x$')
	ax.set_ylabel('Firing Rate $a$ (Hz)')
	plt.show()

def two_template(channel_function):

	#ensemble parameters
	N=50
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.1
	T=0.5
	seed=3

	lif_model=nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref)

	model=nengo.Network(label='Communication Channel')

	with model:
		#stimulus 1 for 0.1<t<0.4 and zero otherwise
		stimulus=nengo.Node(output=lambda t: 0+ 1.0*(0.1<t<0.4))  

		#create ensembles
		ensemble_1=nengo.Ensemble(N,dimensions,
									intercepts=Uniform(-1.0,1.0),
									max_rates=Uniform(100,200),
									neuron_type=lif_model)
		ensemble_2=nengo.Ensemble(N,dimensions,
									intercepts=Uniform(-1.0,1.0),
									max_rates=Uniform(100,200),
									neuron_type=lif_model)

		#connect stimulus to ensemble_1
		stimulation=nengo.Connection(stimulus,ensemble_1)

		#create communication channel between ensemble 1 and 2
		channel=nengo.Connection(ensemble_1,ensemble_2,
									function=channel_function, #identity
									synapse=0.01,  #10ms postsynaptic filter
									solver=LstsqNoise(noise=noise))

		#calculate the 
		#probe the decoded values from the two ensembles
		probe_stim=nengo.Probe(stimulus)
		probe_ensemble_1=nengo.Probe(ensemble_1)
		probe_ensemble_2=nengo.Probe(ensemble_2)

	#run the model
	sim=nengo.Simulator(model,seed=seed)
	sim.run(T)

	#plot inputs and outputs
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(311)
	ax.plot(sim.trange(),sim.data[probe_stim],label='stimulus')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	# ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(312)
	ax.plot(sim.trange(),sim.data[probe_stim],label='input')
	ax.plot(sim.trange(),sim.data[probe_ensemble_1],label='ensemble 1 decoded output')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	# ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(313)
	ax.plot(sim.trange(),sim.data[probe_ensemble_1],label='input')
	ax.plot(sim.trange(),sim.data[probe_ensemble_2],label='ensemble 2 decoded output')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	# ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True)
	plt.tight_layout()
	plt.show()

def two_a():

	channel_function = lambda x: x
	two_template(channel_function)

def two_b():

	channel_function = lambda x: 1.0-2.0*x
	two_template(channel_function)

def three_template(stim_function,neuron_type='spike'):

	#ensemble parameters
	N=200
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	tau_feedback=0.05
	tau_input=0.005
	noise=0.1
	T=1.5
	seed=3

	if neuron_type == 'spike':
		lif_model=nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref)
	elif neuron_type == 'rate':
		lif_model=nengo.LIFRate(tau_rc=tau_rc,tau_ref=tau_ref)

	model=nengo.Network(label='Communication Channel')

	with model:
		stimulus=nengo.Node(output=stim_function)  

		integrator=nengo.Ensemble(N,dimensions,
									intercepts=Uniform(-1.0,1.0),
									max_rates=Uniform(100,200),
									neuron_type=lif_model)

		#define feedforward transformation <=> transform=tau
		def feedforward(u):
			return tau_feedback*u

		stimulation=nengo.Connection(stimulus,integrator,
									function=feedforward,
									# transform=tau_feedback,
									synapse=tau_input)

		#define recurrent transformation
		def recurrent(x):
			return 1.0*x

		#create recurrent connection
		channel=nengo.Connection(integrator,integrator,
									function=recurrent,
									synapse=tau_feedback,  
									solver=LstsqNoise(noise=noise))

		#probes
		probe_stimulus=nengo.Probe(stimulus,synapse=0.01)
		probe_integrator=nengo.Probe(integrator,synapse=0.01)

	#run the model
	sim=nengo.Simulator(model,seed=seed)
	sim.run(T)

	#calculated expected (ideal) using scipy.integrate
	ideal=[integrate.quad(stim_function,0,T)[0] 
		for T in sim.trange()]

	#plot input and integrator value
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(sim.trange(),sim.data[probe_stimulus],label='stimulus')
	ax.plot(sim.trange(),sim.data[probe_integrator],label='integrator')
	ax.plot(sim.trange(),ideal,label='ideal')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	# ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def three_a():

	stim_function = lambda t: 0.9*(0.04<t<1.0)
	neuron_type = 'spike'
	three_template(stim_function,neuron_type)

def three_b():

	stim_function = lambda t: 0.9*(0.04<t<1.0)
	neuron_type = 'rate'
	three_template(stim_function,neuron_type)

def three_c():

	stim_function = lambda t: 0.9*(0.04<t<0.16)
	neuron_type = 'spike'
	three_template(stim_function,neuron_type)

def three_d():

	stim_function = lambda t: 2.0*t*(0.0<=t<=0.45)
	neuron_type = 'spike'
	three_template(stim_function,neuron_type)

def three_e():

	stim_function = lambda t: 5.0*np.sin(5.0*t)
	neuron_type = 'spike'
	three_template(stim_function,neuron_type)

def three_bonus():
    
 	#ensemble parameters
	N=2000
	dimensions=4
	tau_rc=0.02
	tau_ref=0.002
	tau=0.02
	noise=0.001
	T=15.0
	seed=3

	#pendulum parameters
	G = 9.8  # acceleration due to gravity, in m/s^2
	L1 = 1.0  # length of pendulum 1 in m
	L2 = 1.0  # length of pendulum 2 in m
	M1 = 1.0  # mass of pendulum 1 in kg
	M2 = 1.0  # mass of pendulum 2 in kg
    
	# th1 and th2 are the initial angles (degrees)
	# w10 and w20 are the initial angular velocities (degrees per second)
	th1 = 120.0
	w1 = 0.0
	th2 = -10.0
	w2 = 0.0
	t_stim=0.1

	# initial state to push the neurons towards
	state_init = np.radians([th1, w1, th2, w2])
    
	# normal spiking LIF neurons
	lif_model=nengo.LIF(tau_rc=tau_rc,tau_ref=tau_ref)

	model=nengo.Network(label='Double Pendulum')

	with model:

		#push the model into the initial state using stimulus noes
		stimulus_2=nengo.Node(output=lambda t: state_init[1]*(0<t<t_stim))
		stimulus_3=nengo.Node(output=lambda t: state_init[2]*(0<t<t_stim))
		stimulus_4=nengo.Node(output=lambda t: state_init[3]*(0<t<t_stim))
		stimulus_1=nengo.Node(output=lambda t: state_init[0]*(0<t<t_stim))
        
		#create the ensemble
		ens1 = nengo.Ensemble(N,dimensions,
									intercepts=Uniform(-1.0*2.0*np.pi,1.0*2.0*np.pi),
									max_rates=Uniform(100,200),
									neuron_type=lif_model)

		#define recurrent transformation
		def recurrent(state):

			dydx = np.zeros_like(state)
			dydx[0] = state[1]

			del_ = state[2] - state[0]
			den1 = (M1 + M2)*L1 - M2*L1*np.cos(del_)*np.cos(del_)
			dydx[1] = (M2*L1*state[1]*state[1]*np.sin(del_)*np.cos(del_) +
				M2*G*np.sin(state[2])*np.cos(del_) +
				M2*L2*state[3]*state[3]*np.sin(del_) -
				(M1 + M2)*G*np.sin(state[0]))/den1

			dydx[2] = state[3]

			den2 = (L2/L1)*den1
			dydx[3] = (-M2*L2*state[3]*state[3]*np.sin(del_)*np.cos(del_) +
				(M1 + M2)*G*np.sin(state[0])*np.cos(del_) -
				(M1 + M2)*L1*state[1]*state[1]*np.sin(del_) -
				(M1 + M2)*G*np.sin(state[2]))/den2

			return dydx

		#stimulate the ensemble
		stim1=nengo.Connection(stimulus_1,ens1[0],transform=tau,synapse=tau)
		stim2=nengo.Connection(stimulus_2,ens1[1],transform=tau,synapse=tau)
		stim3=nengo.Connection(stimulus_3,ens1[2],transform=tau,synapse=tau)
		stim4=nengo.Connection(stimulus_4,ens1[3],transform=tau,synapse=tau)

		#create recurrent connection
		channel=nengo.Connection(ens1,ens1,
									function=recurrent,
									synapse=tau,  
									solver=LstsqNoise(noise=noise))

		#le probing man
		probe_pendulum=nengo.Probe(ens1,synapse=tau)

	#run the model
	sim=nengo.Simulator(model,seed=seed)
	sim.run(T)

	data=sim.data[probe_pendulum]

	theta1=data[:,0]
	p1=data[:,1]
	theta2=data[:,2]
	p2=data[:,3]

	#transform into x-y coordinates of the respective masses
	x1 = L1*np.sin(data[:, 0])
	y1 = -L1*np.cos(data[:, 0])
	x2 = L2*np.sin(data[:, 2]) + x1
	y2 = -L2*np.cos(data[:, 2]) + y1

	#debugging
	# print theta1

	#plot input and integrator value
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(211)
	ax.plot(sim.trange(),theta1,label='$\\theta_1$')
	ax.plot(sim.trange(),p1,label='$p_1$')
	ax.plot(sim.trange(),theta2,label='$\\theta_2$')
	ax.plot(sim.trange(),p2,label='$p_2$')
	legend=ax.legend(loc='best',shadow=True)
	ax=fig.add_subplot(212)
	ax.plot(x1,y1,label='mass 1')
	ax.plot(x2,y2,label='mass 2')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	# ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True)
	plt.show()
    
def main():
	# one_a()
	# one_b()
	# one_c()
	# one_d()
	# two_a()
	# two_b()
	# three_a()
	# three_b()
	# three_c()
	# three_d()
	# three_e()
	three_bonus()

main()