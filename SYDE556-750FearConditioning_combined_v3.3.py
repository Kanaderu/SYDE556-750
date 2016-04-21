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
filename='FearConditioningMullerCombinedV3pt3'
experiment='muller-tone' #muller-tone, muller-context, viviani
drugs=['saline-saline','muscimol-saline','saline-muscimol','muscimol-muscimol']# ['none','oxytocin'] 
n_trials=2
pairings_train=10 #how many CS-US pairs to train on
tones_test=5
dt=0.001 #timestep
dt_sample=0.01 #probe sample_every
condition_PES_rate = 5e-4 #conditioning learning rate to CS
context_PES_rate = 5e-5 #conditioning learning rate to Context
extinct_PES_rate = 5e-6 #extinction learning rate
gaba_muscimol=1.0 #1.5 -> identical gaba responses, 1.0 -> muscimol-saline = saline-saline
gaba_min=0.2 #minimum amount of inhibition
oxy=0.7 #magnitude of oxytocin stimulus

#ensemble parameters
N=100 #neurons for ensembles
dim=1 #dimensions for ensembles
tau_stim=0.01 #synaptic time constant of stimuli to populations
tau=0.01 #synaptic time constant between ensembles
tau_learn=0.01 #time constant for error populations onto learning rules
tau_drug=0.1 #time constant for application of drugs
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA
tau_recurrent=0.005 #same as GABAergic cells
thresh_error=0.2 #activity in error populations must exceed this value to have futher impact
thresh_inter=0.3 #activity in inhibitory populations must exceed this value to have futher impact
LA_inter_feedback=-0.5 #controlls recurrent inhibition in LA
BA_inter_feedback_F_to_F=0.5 #controls recurrent excitation in BA_fear
BA_inter_feedback_E_to_E=0.5 #controls recurrent excitation in BA_extinct
BA_inter_feedback_F_to_E=-0.2 #controls mutual inhibition b/w BA_fear and BA_extinct
BA_inter_feedback_E_to_F=-0.2 #controls mutual inhibition b/w  BA_extinct and BA_fear

#stimuli
tt=10.0/60.0 #tone time
nt=7.0/60.0 #nothing time #in muller paper nt=9.5/60,st=0.5/60,n2t=0
st=2.0/60.0 #shock time
n2t=1.0/60.0 #nothing time
wt=60.0/60.0 #wait/delay time
t_train=int(pairings_train*(wt+tt)/dt)*dt
t_test=t_train*tones_test/pairings_train #multiply by X/pairings for X tone presentations

params={
	'filename':'FearConditioningCombinedV3',
	'experiment':experiment,
	'drugs':drugs,
	'n_trials':n_trials,
	'pairings_train':pairings_train,
	'tones_test':tones_test,
	'dt':dt,
	'dt_sample':dt_sample,
	'condition_PES_rate':condition_PES_rate,
	'context_PES_rate':context_PES_rate,
	'extinct_PES_rate':extinct_PES_rate,
	'gaba_min':gaba_min,
	'gaba_muscimol':gaba_muscimol,
	'oxy':oxy,

	'N':N,
	'dim':dim,
	'tau_stim':tau_stim,
	'tau':tau,
	'tau_learn':tau_learn,
	'tau_drug':tau_drug,
	'tau_GABA':tau_GABA,
	'tau_Glut':tau_Glut,
	'tau_recurrent':tau_recurrent,
	'thresh_error':thresh_error,
	'thresh_inter':thresh_inter,
    'BA_inter_feedback_F_to_F':BA_inter_feedback_F_to_F,
    'BA_inter_feedback_E_to_E':BA_inter_feedback_E_to_E,
    'BA_inter_feedback_F_to_E':BA_inter_feedback_F_to_E,
    'BA_inter_feedback_E_to_F':BA_inter_feedback_E_to_F,

	'tt':tt,
	'nt':nt,
	'st':st,
	'n2t':n2t,
	'wt':wt,
	't_train':t_train,
	't_test':t_test,
}

'Helper functions and transformations on ensemble connections ########################'''

drug=drugs[0]
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
    if t<t_train and experiment!='viviani': #viviani just has context and US
    	return CS_array[int(t/dt)]
    elif t_train<=t<t_train+t_test and experiment=='muller-tone':
    	return 1 #constant tone
    return 0

def Context_function(t):
    if t<t_train:
    	return 1
    elif t_train<=t<t_train+t_test and experiment=='muller-context':
    	return 1
    elif t_train<=t<t_train+t_test and experiment=='viviani':
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
    if drug=='oxytocin' and t_train<=t<t_train+t_test: 
    	return oxy 
    return 0

#inhibitory interneuron connections out of LA_inter onto LA
def LA_recurrent(x):
	cs=x[:dim] #response to CS, gets learned
	us=x[dim:2*dim]
	inhibit=x[-1]
	feedback=[cs*(LA_inter_feedback-inhibit),us*(LA_inter_feedback-inhibit)]
	return feedback
	
#difference between US and LA activity is used to train CS-LA connection w/o extinction 
def LA_inter_error(x):
    cs=x[:dim]
    us=x[dim:-1]
    inhibit=x[-1]
    error=(us-cs)*max(0,(1-inhibit)) #error signal * inhibitory control, can't go double negative
    return error

#inhibitory interneuron connections out of BA_inter onto BA (through IL/PL/CCK/PV)
def BS_recurrent_F_to_F(x):
	inhibit=x[-3]
	fear=x[-2]
	extinct=x[-1]
	feedback=fear*(BA_inter_feedback_F_to_F-inhibit) #mutual excitation minus gaba inhibition
	return feedback

def BS_recurrent_E_to_E(x):
	inhibit=x[-3]
	fear=x[-2]
	extinct=x[-1]
	feedback=extinct*(BA_inter_feedback_E_to_E-inhibit) #mutual excitation minus gaba inhibition
	return feedback
	
def BS_recurrent_F_to_E(x):
	inhibit=x[-3]
	fear=x[-2]
	extinct=x[-1]
	feedback=fear*(BA_inter_feedback_F_to_E+inhibit) #mutual inhibition minus gaba inhibition
	return feedback
	
def BS_recurrent_E_to_F(x):
	inhibit=x[-3]
	fear=x[-2]
	extinct=x[-1]
	feedback=extinct*(BA_inter_feedback_E_to_F+inhibit) #mutual inhibition minus gaba inhibition
	return feedback
	

#difference between US and LA activity is used to train CS-LA connection w/o extinction 
def BA_inter_error(x):
    context=x[:dim]
    us=x[dim:2*dim]
    inhibit=x[-3]
    error=(us-context)*max(0,(1-inhibit)) #error signal * inhibitory control, can't go double negative
    return error
   
    
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
	stim_motor=nengo.Node(output=1)

	#ENSEMBLES ########################################################################

	#stimulus subpopulations
	U=nengo.Ensemble(N,dim) #intermediary#
	C=nengo.Ensemble(N,dim) #excited by stim_CS
	Context=nengo.Ensemble(N,dim) #excited by stim_CS
	Motor=nengo.Ensemble(N,dim) #indicates movement or freezing

	#Lateral Amygdala subpopulations
	LA=nengo.Ensemble(4*N,2*dim,radius=2) 
	#GABA application targets are local GABAergic interneurons in LA which control 
	#excitability-dependent synaptic plasticity, and therefore fear conditioning,
	#as well as control activity of LA, reducing fear response
	#This population has one extra dimension, "i", which is excited by the GABA stimulus
	LA_inter=nengo.Ensemble(8*N,2*dim+1,radius=2,n_eval_points=3000)
	        
	#Intercalated Cells
	ITCd=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) 
	ITCv=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1))

    #Central Lateral and Central Medial Amygdala subpopulations
	CeL_ON=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeM_DAG=nengo.Ensemble(N,dim) #medial central amygdala, outputs fear responses

	#intra-BA/Cortex/Hippocampus subpopulations
	BA_fear=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) #basolateral amygdala cells activated by extinction
	#BA_inter represent several populations whose exact connections are unknown, and may exist
	#within BA or in nearby hippocampus/cortex.
	#Representation: [context,US,inhibit,Fear_recurrent,Extinct_recurrent]
	#Functions:
	#(a) sustain activity of BA_fear and BA_extinct to produce elongated behavior (integrator->long freeze)
	#(b) mutually inhibit BA_fear and BA_extinct (can't do both at once)
	#(c) provide learning signal for context to BA_fear/BA_extinct populations
	#(d) represent GABAergic activation to allow drug control of (a-c)
	BA_inter=nengo.Ensemble(10*N,2*dim+3,radius=3)
	
	#Error populations
	error_cond=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1))
	error_context=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1))
	error_extinct=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1))

	#CONNECTIONS ########################################################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	nengo.Connection(stim_gaba,LA_inter[-1],synapse=tau_stim) #stimulate the 'control' dimension
	nengo.Connection(stim_oxy,CeL_OFF,synapse=tau_stim)
	
	#Lateral Amygdala connections
	conn_condition=nengo.Connection(C,LA[:dim],synapse=tau_learn,transform=0) #primary learned connection CS-freeze
	nengo.Connection(U,LA[dim:2*dim],synapse=tau) #error signal computed in LA, so it needs US info
	nengo.Connection(LA,LA_inter[:2*dim],synapse=tau_recurrent) #recurrent connection to interneurons
	nengo.Connection(LA_inter,error_cond,synapse=tau_recurrent,function=LA_inter_error)
	nengo.Connection(LA_inter,LA,synapse=tau_recurrent,function=LA_recurrent) #recurrent connection to interneurons
            
    #Basal Nuclei connections, includes possible Cortex/Hippocampus connections
	nengo.Connection(LA[:dim],BA_fear,synapse=tau) #CS-fear circuit
	conn_context=nengo.Connection(Context,BA_fear,synapse=tau,transform=0) #context-fear circuit
	conn_extinct=nengo.Connection(Context,BA_extinct,synapse=tau,transform=0) #context-extinction circuit
	nengo.Connection(Context,BA_inter[:dim],synapse=tau) #context for learning connections
	nengo.Connection(U,BA_inter[dim:2*dim],synapse=tau_stim) #US for learning connections
	nengo.Connection(stim_gaba,BA_inter[-3],synapse=tau_stim) #inhibition for gaba control of learn, recurrent
	nengo.Connection(BA_fear,BA_inter[-2],synapse=tau) #corresponds to known LA to CCK connection
	nengo.Connection(BA_extinct,BA_inter[-1],synapse=tau) #unknown
	nengo.Connection(BA_inter,BA_fear,synapse=tau_recurrent,function=BS_recurrent_F_to_F) #IL
	nengo.Connection(BA_inter,BA_extinct,synapse=tau_recurrent,function=BS_recurrent_E_to_E) #???
	nengo.Connection(BA_inter,BA_extinct,synapse=tau_recurrent,function=BS_recurrent_F_to_E) #CCK
	nengo.Connection(BA_inter,BA_fear,synapse=tau_recurrent,function=BS_recurrent_E_to_F) #PV
	nengo.Connection(BA_inter,error_context,synapse=tau_recurrent,function=BA_inter_error)
	nengo.Connection(BA_inter,error_extinct,synapse=tau_recurrent,transform=-1,function=BA_inter_error)
	
	#Intercalated Cells connections
	nengo.Connection(LA[:dim],CeL_ON,synapse=tau)
	nengo.Connection(LA[:dim],ITCd,synapse=tau) #CeL pathway
	nengo.Connection(BA_extinct,ITCv,synapse=tau)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=tau)

	#Central Lateral and Central Medial Amygdala connections
	nengo.Connection(BA_fear,CeM_DAG,synapse=tau)
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=tau)	
	nengo.Connection(ITCv,CeM_DAG,transform=-1,synapse=tau)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=tau)
	nengo.Connection(CeL_ON,CeM_DAG,synapse=tau_GABA)
	nengo.Connection(CeL_OFF,CeM_DAG,transform=-1)

	#Learning connections
	conn_condition.learning_rule_type=nengo.PES(learning_rate=condition_PES_rate)
	nengo.Connection(error_cond,conn_condition.learning_rule,synapse=tau_learn,transform=-1)
	conn_context.learning_rule_type=nengo.PES(learning_rate=context_PES_rate)
	nengo.Connection(error_context,conn_context.learning_rule,synapse=tau_learn,transform=-1)
	conn_extinct.learning_rule_type=nengo.PES(learning_rate=extinct_PES_rate)
	nengo.Connection(error_extinct,conn_extinct.learning_rule,synapse=tau_learn,transform=-1)
	
	#Motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=tau) #high=movement
	
	
	#PROBES ########################################################################

	motor_probe=nengo.Probe(Motor,synapse=0.01,sample_every=dt_sample)

'''simulation ###############################################'''

columns=('freeze','trial','time','drug')
trials=np.arange(n_trials)
timesteps=np.arange(int(t_train/dt_sample),int((t_train+t_test)/dt_sample))
dataframe = pd.DataFrame(index=np.arange(0, len(drugs)*len(trials)*len(timesteps)),
						columns=columns)

i=0
for drug in drugs:
    for n in trials:
		print 'Running experiment \"%s\", drug \"%s\", trial %s...' %(experiment,drug,n+1)
		sim=nengo.Simulator(model,dt=dt)
		sim.run(t_train+t_test)
		max_motor, min_motor=np.max(sim.data[motor_probe]),np.min(sim.data[motor_probe])
		for t in timesteps:
			freeze=(sim.data[motor_probe][t][0]-max_motor)/(min_motor-max_motor)
			realtime=(t*dt_sample-t_train)*60 #starts at 0 when training ends, units=realtime seconds
			dataframe.loc[i]=[freeze,n,realtime,drug]
			i+=1

'''data analysis, plotting, exporting ###############################################'''

addon=str(np.random.randint(0,100000))
fname=filename+addon

print 'Exporting Data...'
dataframe.to_pickle(fname+'.pkl')
param_df=pd.DataFrame([params])
param_df.reset_index().to_json(fname+'_params.json',orient='records')

print 'Plotting...'
if experiment != 'viviani':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.set(context='paper')
	sns.barplot(x="drug",y="freeze",data=dataframe,ax=ax1)
	sns.tsplot(time="time", value="freeze",
					unit="trial", condition="drug",
					data=dataframe,ax=ax2)
	ax1.set(ylabel='freezing (%)', ylim=(0.0,1.0))
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
else:
	figure, ax1 = plt.subplots(1, 1)
	sns.set(context='paper')
	sns.tsplot(time="time", value="freeze",
					unit="trial", condition="drug",
					data=dataframe,ax=ax1)
	ax1.set(xlabel='time (s)', ylabel='mean(freeze) (%)', ylim=(0.0,1.0))
figure.savefig(fname+'.png')
plt.show()