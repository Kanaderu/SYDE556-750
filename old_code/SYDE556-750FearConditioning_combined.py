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
import pandas as pd
# import ipdb

'''Parameters'''
#simulation parameters
filename='FearConditioningMullerV4'
n_trials=10
pairings_train=5
tones_test=2
dt=0.001 #timestep
dt_sample=0.01 #probe sample_every
experiment='tone' #default, changed in simulation section
drug='saline-saline' #default, changed with gaba_function(t)
condition_PES_rate = 5e-4 #conditioning learning rate to CS
context_PES_rate = 1e-4 #conditioning learning rate to Context
extinct_PES_rate = 5e-5 #extinction learning rate
gaba_muscimol=1.25 #1.5 -> identical gaba responses, 1.0 -> muscimol-saline = saline-saline
oxy=0.7


#ensemble parameters
N=100 #neurons for ensembles
dim=1 #dimensions for ensembles
tau_stim=0.01 #synaptic time constant of stimuli to populations
tau=0.01 #synaptic time constant between ensembles
tau_learn=0.01
tau_drug=0.1
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA
tau_LA_recurrent=0.005 #same as GABAergic cells, could be shorter b/c of locality
thresh_error=0.2
thresh_inter=0.3
gaba_min=0.1
scale_BA_inter=-0.2 #controls integration in BA_fear

#stimuli
tt=10.0/60.0 #tone time
nt=7.0/60.0 #nothing time #experiment nt=9.5/60,st=0.5/60,n2t=0
st=2.0/60.0 #shock time
n2t=1.0/60.0 #nothing time
wt=60.0/60.0 #wait/delay time
t_train=int(pairings_train*(wt+tt)/dt)*dt
t_test=t_train*tones_test/pairings_train #multiply by X/pairings for X tone presentations

params={
	'filename':'FearConditioningMullerV3pt7',
	'n_trials':n_trials,
	'pairings_train':pairings_train,
	'tones_test':tones_test,
	'drug':drug,
	'gaba_muscimol':gaba_muscimol,
	'dt':dt,
	'dt_sample':dt_sample,
	'N':N,
	'dim':dim,
	'tau_stim':tau_stim,
	'tau':tau,
	'condition_PES_rate':condition_PES_rate,
	'context_PES_rate':context_PES_rate,
	'extinct_PES_rate':extinct_PES_rate,
	'tau_learn':tau_learn,
	'tau_drug':tau_drug,
	'tau_GABA':tau_GABA,
	'tau_Glut':tau_Glut,
	'tau_LA_recurrent':tau_LA_recurrent,
	'thresh_error':thresh_error,
	'thresh_inter':thresh_inter,
	'gaba_min':gaba_min,
	'tt':tt,
	'nt':nt,
	'st':st,
	'n2t':n2t,
	'wt':wt,
	't_train':t_train,
	't_test':t_test,
}

'Helper functions and transformations on ensemble connections ########################'''

def make_US_CS_arrays(): #1s sim time = 1min (60s) real time
	rng=np.random.RandomState()
	CS_array=np.zeros((int(t_train/dt)))
	US_array=np.zeros((int(t_train/dt)))
	for i in range(pairings_train):
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
    if drug=='saline-saline': 
    	return gaba_min
    elif drug=='muscimol-saline' and t<t_train:
    	return gaba_muscimol
    elif drug=='saline-muscimol' and t_train<=t<t_train+t_test:
    	return gaba_muscimol
    elif drug=='muscimol-muscimol':
    	return gaba_muscimol
    return gaba_min

def oxy_function(t): #oxytocin activates GABAergic interneurons in CeL_Off
    if drug=='oxytocin' and t_train+t_control<t<t_train+t_control+t_test: 
    	return oxy 
    return 0

def LA_recurrent_in(x):
    cs=x[:dim]
    us=x[dim:]
    return [cs,us]
    
#difference between US and LA activity is used to train CS-LA connection w/o extinction 
def LA_error(x):
    cs=x[:dim]
    us=x[dim:-1]
    inhibit=x[-1]
    error=(1+inhibit)*(us-cs)
    return error

#inhibitory interneuron connections, directly onto LA neurons (bypass encoders)
def LA_recurrent_out(x):
	cs=x[:dim] #response to CS, gets learned
	us=x[dim:2*dim]
	inhibit=x[-1]
	feedback=(cs+us)*(-1.0*inhibit)
	return feedback

#signal used to learn the 'condition' connection (CS-LA)
#threshold need to make signal discernable from noise,
#otherwise initial activation causes runaway learning
def error_out(x):
    if abs(x) > thresh_error:
        return -x
    return 0
    
def BA_inter_feedback(x):
    return (1+scale_BA_inter)*x #controlled integrator
    
'''model definition #################################################'''

model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ########################################################################

	CS_array,US_array=make_US_CS_arrays()
	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)
	stim_Context=nengo.Node(output=Context_function)
	stim_gaba=nengo.Node(output=gaba_function)
	stim_oxy=nengo.Node(output=oxy_function)
	stim_motor=nengo.Node(output=2)

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
	LA_inter=nengo.Ensemble(8*N,2*dim+1,radius=2,n_eval_points=3000, #works
	        encoders=Choice([[1,0,0],[0,1,0],[0,0,1]]),
	        eval_points=Uniform(thresh_inter,1))
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
	BA_inter=nengo.Ensemble(N,dim) #recurrently connected to BA_fear to sustain fear expression
	Context=nengo.Ensemble(N,dim) #excited by stim_CS

	#Error populations
	error_on=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	error_off=nengo.Ensemble(N,dim, encoders=Choice([[-1]]), eval_points=Uniform(-1,0))

	#CONNECTIONS ########################################################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	nengo.Connection(stim_gaba,LA_inter[-1],synapse=tau_stim) #stimulate the 'control' dimension
	nengo.Connection(stim_oxy,CeL_OFF,synapse=tau_stim)
	
	#Amygdala connections
	nengo.Connection(U,LA[dim:2*dim],synapse=tau) #error signal computed in LA, so it needs US info
# 	nengo.Connection(U,LA[dim:2*dim],synapse=1.5*tau,transform=-2) #differentiator
	nengo.Connection(LA,LA_inter[:2*dim],synapse=tau_LA_recurrent,
            function=LA_recurrent_in) #recurrent connection to interneurons
	nengo.Connection(LA_inter,error_on,synapse=tau_LA_recurrent,function=LA_error)
	nengo.Connection(LA_inter,error_off,synapse=tau_LA_recurrent,function=LA_error)
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
	
	#BA-cortex connectionss
	nengo.Connection(BA_fear,BA_inter,synapse=tau)
	nengo.Connection(BA_inter,BA_fear,synapse=tau,function=BA_inter_feedback)
	
	#Motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=tau) #high=movement

	#Learned connections
	condition_PES=nengo.Connection(C,LA[:dim],synapse=tau_learn,transform=0)
	condition_PES.learning_rule_type=nengo.PES(learning_rate=condition_PES_rate)
	nengo.Connection(error_on,condition_PES.learning_rule,synapse=tau_learn,function=error_out)

	context_PES=nengo.Connection(Context,BA_fear,synapse=tau_learn,transform=0)
	context_PES.learning_rule_type=nengo.PES(learning_rate=context_PES_rate)
	nengo.Connection(error_on,context_PES.learning_rule,synapse=tau_learn,function=error_out)

	extinct_PES=nengo.Connection(Context,BA_extinct,synapse=tau_learn,transform=0)
	extinct_PES.learning_rule_type=nengo.PES(learning_rate=extinct_PES_rate)
	nengo.Connection(error_off,extinct_PES.learning_rule,synapse=tau_learn,
	        transform=-1,function=error_out)
	
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

	motor_probe=nengo.Probe(Motor,synapse=0.01,sample_every=dt_sample)

'''simulation ###############################################'''

columns=('motor','trial','time','experiment','drug')
exps=['context']
drugs=['saline-saline','muscimol-saline','saline-muscimol','muscimol-muscimol']
trials=np.arange(n_trials)
timesteps=np.arange(int(t_train/dt_sample),int((t_train+t_test)/dt_sample))
dataframe = pd.DataFrame(index=np.arange(0, len(exps)*len(drugs)*len(trials)*len(timesteps)),
						columns=columns)

i=0
for experiment in exps:
    for drug in drugs:
        for n in trials:
			print 'Running experiment \"%s\", drug \"%s\", trial %s...' %(experiment,drug,n+1)
			sim=nengo.Simulator(model,dt=dt)
			sim.run(t_train+t_test)
			for t in timesteps:
				motor=sim.data[motor_probe][t][0]
				realtime=(t*dt_sample-t_train)*60 #starts at 0 when training ends, units=realtime seconds
				dataframe.loc[i]=[motor,n,realtime,experiment,drug]
				i+=1

'''data analysis, plotting, exporting ###############################################'''

addon=str(np.random.randint(0,100000))
fname=filename+addon

print 'Exporting Data...'
dataframe.to_pickle(fname+'.pkl')
param_df=pd.DataFrame([params])
param_df.reset_index().to_json(fname+'_params.json',orient='records')

print 'Plotting...'
figure, (ax1, ax2) = plt.subplots(2, 1)
sns.set(context='paper')
sns.barplot(x="experiment",y="motor",hue='drug',data=dataframe,ax=ax1)
sns.tsplot(time="time", value="motor",
				unit="trial", condition="drug",
				data=dataframe,ax=ax2)
ax2.set(xlabel='time (s)', ylabel='mean(motor)')
figure.savefig(fname+'.png')
plt.show()