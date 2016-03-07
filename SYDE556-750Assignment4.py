# Peter Duggins
# SYDE 556/750
# March 14, 2015
# Assignment 4

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 24

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
	noise=0.2
	lif_model=nengo.LIFRate(tau_rc=tau_rc,tau_ref=tau_ref)

	#model definition
	model = nengo.Network(label='1D Ensemble of LIF Rate Neurons')
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
	activities_noisy = activities + np.random.normal(
								scale=noise*np.max(activities),
								size=activities.shape)
	decoders = sim.data[connection].weights.T

	#calculate the state estimate
	xhat = np.dot(activities_noisy,decoders)

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
	noise=0.2
	averages=5
	seed=3

	#objects and lists
	radii=np.logspace(-4,8,20)
	RMSE_list=[]
	RMSE_stddev_list=[]

	for r in range(len(radii)):

		RMSE_list_i=[]
		for a in range(averages):

			seed=3+a+r*len(radii)
			rng1=np.random.RandomState(seed=seed)
			lif_model=nengo.LIFRate(tau_rc=tau_rc,tau_ref=tau_ref)

			#generate encoders which are -r OR r
			# encoders = [[radii[r]*(-1+2*rng1.randint(2))] for i in range(N)]
			#generate encoders which are -r TO r
			# encoders = [[radii[r]*(-1+2*rng1.rand())] for i in range(N)]

			#model definition
			model = nengo.Network(label='1D LIF Ensemble',seed=seed)
			with model:
				ens_1d = nengo.Ensemble(N,dimensions,
										intercepts=Uniform(-1.0*r,1.0*r),
										max_rates=Uniform(100,200),
										neuron_type=lif_model)#,encoders=encoders)

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
			decoders = sim.data[connection].weights.T

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
	noise=0.2
	averages=5
	seed=3

	#objects and lists
	tau_refs=np.logspace(-7,-3,5)
	# tau_refs=np.linspace(0.0002,0.0049,10)
	RMSE_list=[]
	RMSE_stddev_list=[]

	for i in range(len(tau_refs)):

		RMSE_list_i=[]
		for a in range(averages):

			seed=3+a+i*len(tau_refs) #unique seed for each iteration
			rng1=np.random.RandomState(seed=seed)
			lif_model=nengo.LIFRate(tau_rc=tau_rc,tau_ref=tau_refs[i])

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
			activities_noisy = activities + rng1.normal(
										scale=noise*np.max(activities),
										size=activities.shape)
			decoders = sim.data[connection].weights.T

			#calculate the state estimate
			xhat = np.dot(activities_noisy,decoders)

			#calculate RMSE
			RMSE_list_i.append(np.sqrt(np.average((eval_points-xhat)**2)))

		#plot tuning curves
		# fig=plt.figure(figsize=(16,8))
		# ax=fig.add_subplot(111)
		# ax.plot(eval_points,activities)
		# ax.set_xlabel('$x$')
		# ax.set_ylabel('Firing Rate $a$ (Hz)')
		# plt.show()
		# ax=fig.add_subplot(212)
		# ax.plot(eval_points,eval_points)
		# ax.plot(eval_points,xhat,
		# 	label='RMSE=%f' %np.average(RMSE_list_i))
		# ax.set_xlabel('$x$')
		# ax.set_ylabel('$\hat{x}$')
		# legend=ax.legend(loc='best',shadow=True)
		# plt.show()

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

def one_d():

	#ensemble parameters
	N=100
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.2
	averages=5
	seed=3

	#objects and lists
	tau_rcs=np.logspace(-3,2,10)
	RMSE_list=[]
	RMSE_stddev_list=[]

	for i in range(len(tau_rcs)):

		RMSE_list_i=[]
		for a in range(averages):

			seed=3+a+i*len(tau_rcs) #unique seed for each iteration
			rng1=np.random.RandomState(seed=seed)
			lif_model=nengo.LIFRate(tau_rc=tau_rcs[i],tau_ref=tau_ref)

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
			activities_noisy = activities + rng1.normal(
										scale=noise*np.max(activities),
										size=activities.shape)
			decoders = sim.data[connection].weights.T

			#calculate the state estimate
			xhat = np.dot(activities_noisy,decoders)

			#calculate RMSE
			RMSE_list_i.append(np.sqrt(np.average((eval_points-xhat)**2)))

		#plot tuning curves and representational accuracy
		# fig=plt.figure(figsize=(16,8))
		# ax=fig.add_subplot(211)
		# ax.plot(eval_points,activities)
		# ax.set_xlabel('$x$')
		# ax.set_ylabel('Firing Rate $a$ (Hz)')
		# ax=fig.add_subplot(212)
		# ax.plot(eval_points,eval_points)
		# ax.plot(eval_points,xhat,
		# 	label='RMSE=%f' %np.average(RMSE_list_i))
		# ax.set_xlabel('$x$')
		# ax.set_ylabel('$\hat{x}$')
		# legend=ax.legend(loc='best',shadow=True)
		# plt.show()

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

def two_a():

	#ensemble parameters
	N=50
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.2
	T=0.5
	seed=3

	lif_model=nengo.LIFRate(tau_rc=tau_rc,tau_ref=tau_ref)

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
									function=lambda x: x, #identity
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
	ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True,fontsize=18)
	ax=fig.add_subplot(312)
	ax.plot(sim.trange(),sim.data[probe_stim],label='input')
	ax.plot(sim.trange(),sim.data[probe_ensemble_1],label='ensemble 1 decoded output')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True,fontsize=18)
	ax=fig.add_subplot(313)
	ax.plot(sim.trange(),sim.data[probe_ensemble_1],label='input')
	ax.plot(sim.trange(),sim.data[probe_ensemble_2],label='ensemble 2 decoded output')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True,fontsize=18)
	plt.tight_layout()
	plt.show()

def two_b():

	#ensemble parameters
	N=50
	dimensions=1
	tau_rc=0.02
	tau_ref=0.002
	noise=0.2
	T=0.5
	seed=3

	lif_model=nengo.LIFRate(tau_rc=tau_rc,tau_ref=tau_ref)

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
									function=lambda x: 1.0-2.0*x,
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
	ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True,fontsize=18)
	ax=fig.add_subplot(312)
	ax.plot(sim.trange(),sim.data[probe_stim],label='input from stimulus')
	ax.plot(sim.trange(),sim.data[probe_ensemble_1],label='ensemble 1 decoded output')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True,fontsize=18)
	ax=fig.add_subplot(313)
	ax.plot(sim.trange(),sim.data[probe_ensemble_1],label='input from ensemble 1')
	ax.plot(sim.trange(),sim.data[probe_ensemble_2],label='ensemble 2 decoded output')
	ax.set_xlabel('time (s)')
	# ax.set_ylabel('value')
	# ax.set_ylim(0,1)
	legend=ax.legend(loc='best',shadow=True,fontsize=18)
	plt.tight_layout()
	plt.show()

def main():
	# one_a()
	# one_b()
	# one_c()
	# one_d()
	# two_a()
	two_b()

main()