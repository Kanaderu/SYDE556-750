# Peter Duggins
# SYDE 556/750
# April 2016
# Final Project - Oxytocin and Fear Conditioning

import nengo
from nengo.dists import Choice,Exponential,Uniform
import nengo_gui
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['font.size'] = 24

#ensemble parameters
stim_N=50 #neurons for stimulus populations
stim_dim=1 #dimensionality of CS and context
N=50 #neurons for ensembles
dim=1 #dimensions for ensembles
tau_stim=0.01 #synaptic time constant of stimuli to populations
tau=0.01 #synaptic time constant between ensembles
condition_rate = 5e-5 #first order conditioning learning rate
extinction_rate = 5e-7 #extinction learning rate
tau_learn=0.01
tau_drug=0.1
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA

#stimuli
t_train=20
t_control=1
t_drug=20
t_extinction=20
dt=0.001
stim_length=0.1

def make_US_array(): #1s sim time = 1min real time
	rng=np.random.RandomState()
	US_array=np.zeros((t_train/dt)) #10 minute training intervals (last 3 min no shock)
	US_times=rng.randint(0,7.0/dt,7) #7 shocks randomly spaced in 7 minutes
	US_times2=rng.randint(10.0/dt,17.0/dt,7)
	for i in US_times:
	    US_array[i:i+stim_length/dt]=1
	for i in US_times2:
	    US_array[i:i+stim_length/dt]=1
	return US_array

def US_function(t):
    if t<t_train: return US_array[int(t/dt)]
    return 0

def CS_function(t):
    return 1

def Context_function(t):
    return CS_function(t)
    
def stop_conditioning_function(t): #learning before testing phase
    if 0<t<t_train: return 0
    return 0

def stop_extinction_function(t): #for testing
    return 0
    
def oxy_function(t):
    if t_train+t_control<t<t_train+t_control+t_drug and subject=='experiment': 
    	return 0.7 #oxytocin application phase
    return 0




'''model definition #################################################'''

model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ####################################
	US_array=make_US_array()
	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)
	stim_Context=nengo.Node(output=Context_function)
	stim_oxy=nengo.Node(output=oxy_function)
	stim_motor=nengo.Node(output=1)
	stop_conditioning=nengo.Node(output=stop_conditioning_function)
	stop_extinction=nengo.Node(output=stop_extinction_function)

	#ENSEMBLES ####################################

	#PAG subpopulations
	U=nengo.Ensemble(N,dim) #intermediary#
    #difference between US and appropriate resopnse (freezing), 0-1 to prevent extinction learning
	Error_ON=nengo.Ensemble(N, dim, encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	Motor=nengo.Ensemble(stim_N,stim_dim) #indicates movement or freezing

	#Amygdala subpopulations
	LA=nengo.Ensemble(N,dim) #lateral amygdala, learns associations
	BA_fear=nengo.Ensemble(N,dim) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(N,dim) #basolateral amygdala cells activated by extinction
	CCK=nengo.Ensemble(N,dim,
	        encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron
	PV=nengo.Ensemble(N,dim,
	        encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron
	ITCd=nengo.Ensemble(N,dim,
        	encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	ITCv=nengo.Ensemble(N,dim,
        	encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	CeL_ON=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeM_DAG=nengo.Ensemble(N,dim) #medial central amygdala, outputs fear responses

	#Cortex/Thalamus subpopulations
	C=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS

	#Hippocampus subpopulations
	Context=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS
	Error_OFF=nengo.Ensemble(N, dim, encoders=Choice([[1]]), eval_points=Uniform(0,1)) #no evidence

	#CONNECTIONS ####################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	nengo.Connection(stim_oxy,CeL_OFF,synapse=tau_drug)
	
	#Amygdala connections
	nengo.Connection(LA,BA_fear,synapse=tau) #LA pathway: normal fear circuit
	nengo.Connection(BA_fear,CeM_DAG,synapse=tau)

	nengo.Connection(LA,ITCd,synapse=tau) #CeL pathway: oxytocin modulated
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=tau)
	nengo.Connection(LA,CeL_ON,synapse=tau)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=tau)
	nengo.Connection(CeL_ON,CeM_DAG,synapse=tau_GABA)
	nengo.Connection(CeL_OFF,CeM_DAG,transform=-1)
	
	nengo.Connection(LA,CCK,synapse=tau) #BA pathway: extinction circuit
	nengo.Connection(CCK,BA_extinct,transform=-1,synapse=tau)
	nengo.Connection(BA_extinct,ITCv,synapse=tau)
	nengo.Connection(ITCv,CeM_DAG,transform=-1,synapse=tau)
	nengo.Connection(BA_fear,CCK,synapse=tau)
	nengo.Connection(BA_extinct,PV,synapse=tau)
	nengo.Connection(PV,BA_fear,transform=-1,synapse=tau)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=tau)
	
	#motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=tau)

	#Learned connections
	conditioning=nengo.Connection(C,LA,function=lambda x: [0]*dim,synapse=tau_learn)
	extinction=nengo.Connection(Context,BA_extinct,function=lambda x: [0]*dim,synapse=tau_learn)
	conditioning.learning_rule_type=nengo.PES(learning_rate=condition_rate)
	extinction.learning_rule_type=nengo.PES(learning_rate=extinction_rate)
	
	#Error calculations
	nengo.Connection(Error_ON, conditioning.learning_rule, transform=-1)
	nengo.Connection(U,Error_ON,transform=1,synapse=tau)
	nengo.Connection(CeM_DAG, Error_ON,transform=-1,synapse=tau_learn)

	nengo.Connection(Error_OFF, extinction.learning_rule, transform=-0.5)
	nengo.Connection(U, Error_OFF,transform=-1,synapse=tau_learn)
	nengo.Connection(CeM_DAG, Error_OFF,transform=1,synapse=tau_learn)

	nengo.Connection(stop_conditioning, Error_ON.neurons, transform=-10*np.ones((N, dim)))
	nengo.Connection(stop_extinction, Error_OFF.neurons, transform=-10*np.ones((N, dim)))

	#PROBES ####################################
	CeM_DAG_voltage=nengo.Probe(CeM_DAG.neurons,'voltage')
	motor_probe=nengo.Probe(Motor,synapse=0.01)






'''simulation and data plotting ###############################################
Try to reproduce figure S2B from Viviani et al (2011)'''

n_trials=10
freezing_control_list=[]
freezing_oxy_list=[]
freezing_extinction_list=[]

for i in range(n_trials):
	subject='experiment'
	print 'Running %s trial %s...' %(subject,i)
	sim=nengo.Simulator(model)
	sim.run(t_train+t_control+t_drug+1+t_extinction)

	motor_value_control=sim.data[motor_probe][t_train/dt:(t_train+t_control)/dt]
	motor_value_oxy=sim.data[motor_probe][(t_train+t_control)/dt:(t_train+t_control+t_drug)/dt]
	motor_value_extinct=sim.data[motor_probe][(t_train+t_control+t_drug+1)/dt:(t_train+t_control+t_drug+1+t_extinction)/dt]

	freezing_control_list.append(1.0-1.0*np.average(motor_value_control))
	freezing_oxy_list.append(1.0-1.0*np.average(motor_value_oxy))
	freezing_extinction_list.append(1.0-1.0*np.average(motor_value_extinct))

avg_freezing_control=np.average(freezing_control_list)
avg_freezing_oxy=np.average(freezing_oxy_list)
avg_freezing_extinction=np.average(freezing_extinction_list)
std_freezing_control=np.std(freezing_control_list)
std_freezing_oxy=np.std(freezing_oxy_list)
std_freezing_extinction=np.std(freezing_extinction_list)


n_trials=10
freezing_control_list_2=[]
freezing_oxy_list_2=[]
freezing_extinction_list_2=[]

for i in range(n_trials):
	subject='control'
	print 'Running %s trial %s...' %(subject,i)
	sim=nengo.Simulator(model)
	sim.run(t_train+t_control+t_drug+1+t_extinction)

	motor_value_control=sim.data[motor_probe][t_train/dt:(t_train+t_control)/dt]
	motor_value_oxy=sim.data[motor_probe][(t_train+t_control)/dt:(t_train+t_control+t_drug)/dt]
	motor_value_extinct=sim.data[motor_probe][(t_train+t_control+t_drug+1)/dt:(t_train+t_control+t_drug+1+t_extinction)/dt]

	freezing_control_list_2.append(1.0-1.0*np.average(motor_value_control))
	freezing_oxy_list_2.append(1.0-1.0*np.average(motor_value_oxy))
	freezing_extinction_list_2.append(1.0-1.0*np.average(motor_value_extinct))

avg_freezing_control_2=np.average(freezing_control_list_2)
avg_freezing_oxy_2=np.average(freezing_oxy_list_2)
avg_freezing_extinction_2=np.average(freezing_extinction_list_2)
std_freezing_control_2=np.std(freezing_control_list_2)
std_freezing_oxy_2=np.std(freezing_oxy_list_2)
std_freezing_extinction_2=np.std(freezing_extinction_list_2)

#Bar Plots
height_freezing=[avg_freezing_control,avg_freezing_oxy,avg_freezing_extinction]
std_freezing=[std_freezing_control,std_freezing_oxy,std_freezing_extinction]
height_freezing_2=[avg_freezing_control_2,avg_freezing_oxy_2,avg_freezing_extinction_2]
std_freezing_2=[std_freezing_control_2,std_freezing_oxy_2,std_freezing_extinction_2]

fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(111)
ax.bar(np.arange(len(height_freezing_2)),height_freezing_2,
		width=0.33,yerr=std_freezing,label='control',color='b')
ax.bar(np.arange(len(height_freezing))+0.33,height_freezing,
		width=0.33,yerr=std_freezing,label='experiment',color='g')
legend=ax.legend(loc='best',shadow=True)
ax.set_xticks([.5,1.5,2.5])
ax.set_xticklabels(('post-training', 'oxytocin applied', 'post-extinction'))
ax.set_ylabel('Freezing')
plt.show()