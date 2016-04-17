# Peter Duggins
# SYDE 556/750
# April 2016
# Final Project - Oxytocin and Fear Conditioning

import nengo
from nengo.dists import Choice,Exponential,Uniform
import nengo_gui
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb

#ensemble parameters
N=100 #neurons for ensembles
dim=1 #dimensions for ensembles
tau_stim=0.01 #synaptic time constant of stimuli to populations
tau=0.01 #synaptic time constant between ensembles
condition_PES_rate = 5e-4 #first order conditioning learning rate
condition_BCM_rate = 5e-10 #first order conditioning learning rate
extinction_rate = 5e-7 #extinction learning rate
tau_learn=0.01
tau_drug=0.1
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA
tau_LA_recurrent=0.005 #same as GABAergic cells, could be shorter b/c of locality

#stimuli
dt=0.001 #timestep
tt=10.0/60.0 #tone time
nt=7.0/60.0 #nothing time #experimeng nt=9.5/60,st=0.5/60,n2t=0
st=2.0/60.0 #shock time
n2t=1.0/60.0 #nothing time
wt=1.0 #wait/delay time
pairings=10
t_train=int(pairings*(wt+tt)/dt)*dt
t_test=t_train/pairings
subject='saline-saline'
gaba_min=0.0
gaba_muscimol=1.0
T_error=0.25

def make_US_CS_arrays(): #1s sim time = 1min (60s) real time
	rng=np.random.RandomState()
	CS_array=np.zeros((int(t_train/dt)))
	US_array=np.zeros((int(t_train/dt)))
	for i in range(pairings):
		CS_array[i*(wt+tt)/dt : (i*(wt+tt)+tt)/dt]=1 # tone
		US_array[i*(wt+tt)/dt : (i*(wt+tt)+nt)/dt]=0 # nothing
		US_array[(i*(wt+tt)+nt)/dt : (i*(wt+tt)+nt+st)/dt]=2 # shock
		US_array[(i*(wt+tt)+nt+st)/dt : (i*(wt+tt)+nt+st+n2t)/dt]=0 # nothing
		CS_array[(i*(wt+tt)+tt)/dt : (i+1)*(wt+tt)/dt]=0 # delay
		US_array[(i*(wt+tt)+tt)/dt : (i+1)*(wt+tt)/dt]=0 # delay
	return CS_array,US_array 

def US_function(t):
    if t<t_train:
    	return US_array[int(t/dt)]
    return 0

def CS_function(t):
    if t<t_train:
    	return CS_array[int(t/dt)]
    # elif t_train<=t<t_train+t_test and experiment=='tone':
    # 	return CS_array[int((t-t_train)/dt)] #spaced out tones, as in training
    elif t_train<=t<t_train+t_test and experiment=='tone':
    	return 1 #constant tone
    return 0

def Context_function(t):
    if t<t_train:
    	return 1
    elif t_train<=t<t_train+t_test and experiment=='context':
    	return 1
    return 0

def gaba_function(t): #activate GABA receptors in LA => inhibition of LA => no learning
    if subject=='saline-saline': 
    	return gaba_min
    elif subject=='muscimol-saline' and t<t_train:
    	return gaba_muscimol
    elif subject=='saline-muscimol' and t_train<=t<t_train+t_test:
    	return gaba_muscimol
    elif subject=='muscimol-muscimol':
    	return gaba_muscimol
    return gaba_min

def LA_recurrent_in(x):
    cs=x[:dim]
    us=x[dim:]
    #need to make these discernable from noise, otherwise initial activation causes runaway learning
    if abs(cs) < T_error*np.ones((dim)):
        cs=np.zeros((dim))
    if abs(us) < T_error*np.ones((dim)):
        us=np.zeros((dim))
    return [cs,us]
    
#difference between US and LA activity is used to train CS-LA connection w/o extinction 
def LA_error(x):
    cs=x[:dim]
    us=x[dim:-1]
    inhibit=x[-1]
    error=np.maximum(np.zeros((dim)),us-cs)
    return error

#inhibitory interneuron connections, directly onto LA neurons (bypass encoders)
def LA_recurrent_out(x):
	cs=x[:dim] #response to CS, gets learned
	us=x[dim:2*dim]
	inhibit=x[-1]
	feedback=(cs+us)*(-1.0*inhibit)
	return feedback

def error_out(x):
    if x > T_error:
        return -x
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

	#ENSEMBLES ########################################################################

	#PAG subpopulations
	U=nengo.Ensemble(N,dim) #intermediary#
	Motor=nengo.Ensemble(N,dim) #indicates movement or freezing

	#Amygdala subpopulations
	#lateral amygdala, learns associations b/w CS and US (no extinction)
	LA=nengo.Ensemble(4*N,2*dim,radius=2) 
	#GABA application targets are local GABAergic interneurons in LA which control 
	#excitability-dependent synaptic plasticity, and therefore fear conditioning,
	#as well as control activity of LA, reducing fear response
	#This population has one extra dimension, "i", which is excited by the GABA stimulus
	LA_inter=nengo.Ensemble(4*N,2*dim+1,radius=2,n_eval_points=3000)
	error_cond=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	BA_fear=nengo.Ensemble(N,dim) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(N,dim) #basolateral amygdala cells activated by extinction
	CCK=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron 1
	PV=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron 2
	ITCd=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	ITCv=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	CeL_ON=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeM_DAG=nengo.Ensemble(N,dim) #medial central amygdala, outputs fear responses

	#Cortex/Thalamus subpopulations
	C=nengo.Ensemble(N,dim) #excited by stim_CS

	#Hippocampus subpopulations
	Context=nengo.Ensemble(N,dim) #excited by stim_CS
	# Error_OFF=nengo.Ensemble(N, dim, encoders=Choice([[1]]), eval_points=Uniform(0,1)) #no evidence

	#CONNECTIONS ########################################################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	nengo.Connection(stim_gaba,LA_inter[-1],synapse=tau_stim) #stimulate the 'control' dimension
	
	#Amygdala connections
	nengo.Connection(U,LA[dim:2*dim],synapse=tau) #error signal computed in LA, so it needs US info
# 	nengo.Connection(U,LA[dim:2*dim],synapse=1.5*tau,transform=-2) #differentiator
	nengo.Connection(LA,LA_inter[:2*dim],synapse=tau_LA_recurrent,
            function=LA_recurrent_in) #recurrent connection to interneurons
	nengo.Connection(LA_inter,error_cond,synapse=tau_LA_recurrent,function=LA_error)
	nengo.Connection(LA_inter,LA.neurons,synapse=tau_LA_recurrent,
            function=LA_recurrent_out,transform=np.ones((4*N,1))) #recurrent connection to interneurons
	nengo.Connection(LA[:dim],BA_fear,synapse=tau) #LA pathway: normal fear circuit
	nengo.Connection(BA_fear,CeM_DAG,synapse=tau)
	nengo.Connection(LA[:dim],ITCd,synapse=tau) #CeL pathway
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=tau)
	nengo.Connection(LA[:dim],CeL_ON,synapse=tau)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=tau)
	nengo.Connection(CeL_ON,CeM_DAG,synapse=tau_GABA)
	nengo.Connection(CeL_OFF,CeM_DAG,transform=-1)
	nengo.Connection(LA[:dim],CCK,synapse=tau) #BA pathway: extinction circuit
	nengo.Connection(CCK,BA_extinct,transform=-1,synapse=tau)
	nengo.Connection(BA_extinct,ITCv,synapse=tau)
	nengo.Connection(ITCv,CeM_DAG,transform=-1,synapse=tau)
	nengo.Connection(BA_fear,CCK,synapse=tau)
	nengo.Connection(BA_extinct,PV,synapse=tau)
	nengo.Connection(PV,BA_fear,transform=-1,synapse=tau)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=tau)
	
	#Motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=tau) #high=movement

	#Learned connections
	condition_PES=nengo.Connection(C,LA[:dim],synapse=tau_learn,transform=0)
	condition_PES.learning_rule_type=nengo.PES(learning_rate=condition_PES_rate)
# 	nengo.Connection(LA_inter,condition_PES.learning_rule,synapse=tau_learn,function=LA_error)
	nengo.Connection(error_cond,condition_PES.learning_rule,synapse=tau_learn,function=error_out)

# 	condition_BCM=nengo.Connection(C,LA[:dim],synapse=tau_learn,
# 	        function=lambda x: x,
#             solver=nengo.solvers.LstsqL2(weights=True))
# 	condition_BCM.learning_rule_type=nengo.BCM(learning_rate=condition_BCM_rate)
# 	condition_both=nengo.Connection(C,LA[:dim],synapse=tau_learn,
# 	        function=lambda x: np.random.random(dim),
# 	        solver=nengo.solvers.LstsqL2(weights=True))
# 	condition_both.learning_rule_type=[nengo.PES(learning_rate=condition_PES_rate),
#         	nengo.BCM(learning_rate=condition_BCM_rate)]
	# extinction=nengo.Connection(Context,BA_extinct,function=lambda x: [0]*dim,synapse=tau_learn)
	# extinction.learning_rule_type=nengo.PES(learning_rate=extinction_rate)
	
	#Error calculations
	# nengo.Connection(Error_OFF, extinction.learning_rule, transform=-1)
	# nengo.Connection(U, Error_OFF,transform=-1,synapse=tau_learn)
	# nengo.Connection(CeM_DAG, Error_OFF,transform=1,synapse=tau_learn)

	#PROBES ########################################################################

	motor_probe=nengo.Probe(Motor,synapse=0.01)






'''simulation and data plotting ###############################################
Try to reproduce figure 3 from Muller et al (2007)'''

n_trials=10

avg_freezing={}
std_freezing={}
freezing={}
avg_timeseries={}
std_timeseries={}
timeseries={}
for exp in ['tone']:#,'context']:
	experiment=exp
	avg_freezing[experiment]={}
	std_freezing[experiment]={}
	avg_timeseries[experiment]={}
	std_timeseries[experiment]={}
	for subj in ['saline-saline', 'muscimol-saline', 'saline-muscimol', 'muscimol-muscimol']:
		subject=subj
		freezing[subject]=[]
		timeseries[subject]=[]
		for i in range(n_trials):
			print 'Running group \"%s\" drug \"%s\" trial %s...' %(experiment,subject,i+1)
			sim=nengo.Simulator(model)
			sim.run(t_train+t_test)
			motor_values=sim.data[motor_probe][int(t_train/dt):int((t_train+t_test)/dt)]
			timeseries[subject].append(motor_values)
			freezing[subject].append(np.average(motor_values))
		avg_timeseries[experiment][subject]=np.average(timeseries[subject],axis=0)
		std_timeseries[experiment][subject]=np.std(timeseries[subject],axis=0)		
		avg_freezing[experiment][subject]=np.average(freezing[subject])
		std_freezing[experiment][subject]=np.std(freezing[subject])

#Bar Plots
avg_tone=[avg_freezing['tone']['saline-saline'],avg_freezing['tone']['muscimol-saline'],
	avg_freezing['tone']['saline-muscimol'],avg_freezing['tone']['muscimol-muscimol']]
std_tone=[std_freezing['tone']['saline-saline'],std_freezing['tone']['muscimol-saline'],
	std_freezing['tone']['saline-muscimol'],std_freezing['tone']['muscimol-muscimol']]
# avg_context=[avg_freezing['context']['saline-saline'],avg_freezing['context']['muscimol-saline'],
# 	avg_freezing['context']['saline-muscimol'],avg_freezing['context']['muscimol-muscimol']]
# std_context=[std_freezing['context']['saline-saline'],std_freezing['context']['muscimol-saline'],
# 	avg_freezing['context']['saline-muscimol'],avg_freezing['context']['muscimol-muscimol']]

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8)) #, sharex=True

ax1.bar(np.arange(len(avg_tone))+0.25,avg_tone,
		width=0.5,yerr=std_tone,label='tone') #,color='b',ecolor='k'
ax1.set_xticks([0.5,1.5,2.5,3.5])
ax1.set_xticklabels(('saline\nsaline', 'muscimol\nsaline','saline\nmuscimol', 'muscimol\nmuscimol'))
ax1.set_ylabel('Freezing')

x_range=sim.trange()[int(t_train/dt):int((t_train+t_test)/dt)]
y1=np.array(avg_timeseries['tone']['saline-saline']).ravel()
e1=np.array(std_timeseries['tone']['saline-saline']).ravel()
ax2.plot(x_range,y1,label='saline-saline')
ax2.fill_between(x_range,y1-e1,y1+e1,color='0.2')
y2=np.array(avg_timeseries['tone']['muscimol-saline']).ravel()
e2=np.array(std_timeseries['tone']['muscimol-saline']).ravel()
ax2.plot(x_range,y2,label='muscimol-saline')
ax2.fill_between(x_range,y2-e2,y2+e2,color='0.4')
y3=np.array(avg_timeseries['tone']['saline-muscimol']).ravel()
e3=np.array(std_timeseries['tone']['saline-muscimol']).ravel()
ax2.plot(x_range,y3,label='saline-muscimol')
ax2.fill_between(x_range,y3-e3,y3+e3,color='0.6')
y4=np.array(avg_timeseries['tone']['muscimol-muscimol']).ravel()
e4=np.array(std_timeseries['tone']['muscimol-muscimol']).ravel()
ax2.plot(x_range,y4,label='muscimol-muscimol')
ax2.fill_between(x_range,y4-e4,y4+e4,color='0.8')
ax2.set_xlabel('time')
ax2.set_ylabel('motor decoded value')
legend=ax2.legend(loc='best',shadow=True)

# ax2.plot(x_range,avg_timeseries['tone']['muscimol-saline'],
# 	label='muscimol-saline',yerr=std_timeseries['tone']['muscimol-saline'])
# ax2.plot(x_range,avg_timeseries['tone']['saline-muscimol'],
# 	label='saline-muscimol',yerr=std_timeseries['tone']['saline-muscimol'])
# ax2.plot(x_range,avg_timeseries['tone']['muscimol-muscimol'],
# 	label='muscimol-muscimol',yerr=std_timeseries['tone']['muscimol-muscimol'])

# ax=fig.add_subplot(122)
# ax.bar(np.arange(len(avg_context))+0.25,avg_context,
# 		width=0.5,yerr=std_context,label='tone',color='g',ecolor='k')
# plt.title('Context')
# ax.set_xticks([0.5,1.5,2.5,3.5])
# ax.set_xticklabels(('saline\nsaline', 'muscimol\nsaline','saline\nmuscimol', 'muscimol\nmuscimol'))
# ax.set_ylabel('Freezing')
plt.tight_layout()
plt.show()