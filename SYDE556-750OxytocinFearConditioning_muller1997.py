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
ens_N=50 #neurons for ensembles
ens_dim=1 #dimensions for ensembles
tau_stim=0.01 #synaptic time constant of stimuli to populations
ens_syn=0.01 #synaptic time constant between ensembles
condition_rate = 5e-5 #first order conditioning learning rate
extinction_rate = 5e-7 #extinction learning rate
learn_syn=0.01
tau_drug=0.1
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA

#stimuli
dt=0.001
tt=10.0/60.0
nt=9.5/60.0
st=0.5/60.0
wt=1.0
t_train=int(5*(wt+tt)/dt)*dt
t_test=t_train
subject='saline-saline'

def make_US_CS_arrays(): #1s sim time = 1min (60s) real time
	rng=np.random.RandomState()
	CS_array=np.zeros((int(t_train/dt)))
	US_array=np.zeros((int(t_train/dt)))
	for i in range(5):
		print i, i*(wt+tt)/dt, (i+1)*(tt)/dt
		CS_array[i*(wt+tt)/dt : (i+1)*(tt)/dt]=1 #10 sec of tone
		print CS_array[i*(wt+tt)/dt : (i+1)*(tt)/dt]
		US_array[i*(wt+tt)/dt : (i+1)*(nt)/dt]=0 #9.5 sec of nothing
		US_array[(i+1)*(nt)/dt : (i+1)*(tt)/dt]=1 #0.5 sec of shock
		CS_array[(i+1)*(tt)/dt : (i+1)*(wt+tt)/dt]=0 #1 min delay
		US_array[(i+1)*(tt)/dt : (i+1)*(wt+tt)/dt]=0 #1 min delay
	print CS_array.sum()
	return CS_array,US_array 

def US_function(t):
    if t<t_train:
    	return US_array[int(t/dt)]
    return 0

def CS_function(t):
    if t<t_train:
    	return CS_array[int(t/dt)]
    elif t_train<=t<t_train+t_test and experiment=='tone':
    	return CS_array[int((t-t_train)/dt)]
    return 0

def Context_function(t):
    if t<t_train:
    	return 1
    elif t_train<=t<t_train+t_test and experiment=='context':
    	return 1
    return 0

def gaba_function(t):
    if subject=='saline-saline': 
    	return 0.0
    elif subject=='muscimol-saline' and t<t_train:
    	return 1.0
    elif subject=='saline-muscimol' and t_train<=t<t_train+t_test:
    	return 1.0    	
    elif subject=='muscimol-muscimol':
    	return 1.0   
    return 0

def stop_conditioning_function(t):
    return 0

def stop_extinction_function(t):
    return 0





'''model definition #################################################'''

model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ########################################################################

	CS_array,US_array=make_US_CS_arrays()
	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)
	stim_Context=nengo.Node(output=Context_function)
	stim_gaba=nengo.Node(output=gaba_function)
	stim_motor=nengo.Node(output=1)
	stop_conditioning=nengo.Node(output=stop_conditioning_function)
	stop_extinction=nengo.Node(output=stop_extinction_function)

	#ENSEMBLES ########################################################################

	#PAG subpopulations
	U=nengo.Ensemble(ens_N,ens_dim) #intermediary#
    #difference between US and appropriate resopnse (freezing), 0-1 to prevent extinction learning
	Error_ON=nengo.Ensemble(ens_N, ens_dim, encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	Motor=nengo.Ensemble(stim_N,stim_dim) #indicates movement or freezing

	#Amygdala subpopulations
	LA=nengo.Ensemble(ens_N,ens_dim) #lateral amygdala, learns associations
	BA_fear=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala cells activated by extinction
	BA_int1=nengo.Ensemble(ens_N,ens_dim,
	        encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron 1
	BA_int2=nengo.Ensemble(ens_N,ens_dim,
	        encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron 2
	ITCd=nengo.Ensemble(ens_N,ens_dim,
        	encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	ITCv=nengo.Ensemble(ens_N,ens_dim,
        	encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	CeL_ON=nengo.Ensemble(ens_N,ens_dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(ens_N,ens_dim) #ON cells in the lateral central amygdala
	CeM_DAG=nengo.Ensemble(ens_N,ens_dim) #medial central amygdala, outputs fear responses

	#Cortex/Thalamus subpopulations
	C=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS

	#Hippocampus subpopulations
	Context=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS
	Error_OFF=nengo.Ensemble(ens_N, ens_dim, encoders=Choice([[1]]), eval_points=Uniform(0,1)) #no evidence

	#CONNECTIONS ########################################################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	#GABA application targets are local GABAergic interneurons in LA which control 
	#excitability-dependent synaptic plasticity, and therefore fear conditioning
	#(Duvarci and Pare 2014, Figure 1 Muller et al 1997)
	nengo.Connection(stim_gaba,LA,transform=-1) #simple approximation; next try recurrent network?
	# nengo.Connection(stim_gaba,CeM_DAG,synapse=tau_drug)
	
	#Amygdala connections
	nengo.Connection(LA,BA_fear,synapse=ens_syn) #LA pathway: normal fear circuit
	nengo.Connection(BA_fear,CeM_DAG,synapse=ens_syn)
	nengo.Connection(LA,ITCd,synapse=ens_syn) #CeL pathway
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(LA,CeL_ON,synapse=ens_syn)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(CeL_ON,CeM_DAG,synapse=tau_GABA)
	nengo.Connection(CeL_OFF,CeM_DAG,transform=-1)
	nengo.Connection(LA,BA_int1,synapse=ens_syn) #BA pathway: extinction circuit
	nengo.Connection(BA_int1,BA_extinct,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_extinct,ITCv,synapse=ens_syn)
	nengo.Connection(ITCv,CeM_DAG,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_fear,BA_int1,synapse=ens_syn)
	nengo.Connection(BA_extinct,BA_int2,synapse=ens_syn)
	nengo.Connection(BA_int2,BA_fear,transform=-1,synapse=ens_syn)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=ens_syn)
	
	#motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=ens_syn)

	#Learned connections
	conditioning=nengo.Connection(C,LA,function=lambda x: [0]*ens_dim,synapse=learn_syn)
	extinction=nengo.Connection(Context,BA_extinct,function=lambda x: [0]*ens_dim,synapse=learn_syn)
	conditioning.learning_rule_type=nengo.PES(learning_rate=condition_rate)
	extinction.learning_rule_type=nengo.PES(learning_rate=extinction_rate)
	
	#Error calculations
	nengo.Connection(Error_ON, conditioning.learning_rule, transform=-1)
	nengo.Connection(U,Error_ON,transform=1,synapse=ens_syn)
	nengo.Connection(CeM_DAG, Error_ON,transform=-1,synapse=learn_syn)
	nengo.Connection(Error_OFF, extinction.learning_rule, transform=-0.5)
	nengo.Connection(U, Error_OFF,transform=-1,synapse=learn_syn)
	nengo.Connection(CeM_DAG, Error_OFF,transform=1,synapse=learn_syn)
	nengo.Connection(stop_conditioning, Error_ON.neurons, transform=-10*np.ones((ens_N, ens_dim)))
	nengo.Connection(stop_extinction, Error_OFF.neurons, transform=-10*np.ones((ens_N, ens_dim)))

	#PROBES ########################################################################

	CeM_DAG_voltage=nengo.Probe(CeM_DAG.neurons,'voltage')
	motor_probe=nengo.Probe(Motor,synapse=0.01)






'''simulation and data plotting ###############################################
Try to reproduce figure 3 from Muller et al (2007)'''

n_trials=2
freezing=[]
for i in range(n_trials):
	experiment='tone'
	subject='saline-saline'
	print 'Running group \"%s\" drug \"%s\" trial %s...' %(experiment,subject,i)
	sim=nengo.Simulator(model)
	sim.run(t_train+t_test)
	motor_value=sim.data[motor_probe][int(t_train/dt):int((t_train+t_test)/dt)]
	freezing.append(1.0-1.0*np.average(motor_value))
avg_freezing=np.average(freezing)
std_freezing=np.std(freezing)

#Bar Plots
height_freezing=[avg_freezing]
std_freezing=[std_freezing]

fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(111)
ax.bar(np.arange(len(height_freezing)),height_freezing,
		width=0.33,yerr=std_freezing,label='control',color='b')
# ax.bar(np.arange(len(height_freezing))+0.33,height_freezing,
# 		width=0.33,yerr=std_freezing,label='experiment',color='g')
legend=ax.legend(loc='best',shadow=True)
ax.set_xticks([.5,1.5,2.5,3.5])
ax.set_xticklabels(('sal-sal', 'musc-sal','sal-musc', 'musc-musc'))
ax.set_ylabel('Freezing')
plt.title('Tone')
plt.show()