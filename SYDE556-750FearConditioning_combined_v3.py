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
filename='FearConditioningMullerCombinedV2'
experiment='viviani' #muller-tone, muller-context, viviani
drugs=['none','oxytocin'] #['saline-saline','muscimol-saline','saline-muscimol','muscimol-muscimol']
n_trials=10
pairings_train=10
tones_test=5
dt=0.001 #timestep
dt_sample=0.01 #probe sample_every
condition_PES_rate = 5e-4 #conditioning learning rate to CS
context_PES_rate = 5e-5 #conditioning learning rate to Context
extinct_PES_rate = 5e-6 #extinction learning rate
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
tau_recurrent=0.005 #same as GABAergic cells, could be shorter b/c of locality
thresh_error=0.2
thresh_inter=0.3
gaba_min=0.2
BA_inter_feedback_excite=0.0 #controls integration in BA_fear: -1=damp,0=none,1=integrate
BA_inter_feedback_inhibit=-1.0 #controls mutual inhibition b/w BA_fear and BA_excite

#stimuli
tt=10.0/60.0 #tone time
nt=7.0/60.0 #nothing time #in muller paper nt=9.5/60,st=0.5/60,n2t=0
st=2.0/60.0 #shock time
n2t=1.0/60.0 #nothing time
wt=60.0/60.0 #wait/delay time
t_train=int(pairings_train*(wt+tt)/dt)*dt
t_test=t_train*tones_test/pairings_train #multiply by X/pairings for X tone presentations

params={
	'filename':'FearConditioningMullerV3pt7',
	'experiment':experiment,
	'drugs':drugs,
	'n_trials':n_trials,
	'pairings_train':pairings_train,
	'tones_test':tones_test,
	'gaba_muscimol':gaba_muscimol,
	'oxy':oxy,
	'dt':dt,
	'dt_sample':dt_sample,
	'condition_PES_rate':condition_PES_rate,
	'context_PES_rate':context_PES_rate,
	'extinct_PES_rate':extinct_PES_rate,

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
	'gaba_min':gaba_min,
	'BA_inter_feedback_excite':BA_inter_feedback_excite,
	'BA_inter_feedback_inhibit':BA_inter_feedback_inhibit,

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

#inhibitory interneuron connections, directly onto LA neurons (bypass encoders)
def LA_recurrent_out(x):
	cs=x[:dim] #response to CS, gets learned
	us=x[dim:2*dim]
	inhibit=x[-1]
	feedback=[cs*(-1.0*inhibit),us*(-1.0*inhibit)]
	return feedback
	
#difference between US and LA activity is used to train CS-LA connection w/o extinction 
def LA_inter_error(x):
    cs=x[:dim]
    us=x[dim:-1]
    inhibit=x[-1]
    error=(1+inhibit)*(us-cs)
    return error
    
#difference between US and LA activity is used to train CS-LA connection w/o extinction 
def BA_inter_error(x):
    context=x[:dim]
    us=x[dim:2*dim]
    inhibit=x[-3]
    error=(1+inhibit)*(us-context)
    return error

#signal used to learn the conditioning, context, and extinction connections
#threshold need to make signal discernable from noise,
#otherwise initial activation causes runaway learning
def error_out(x):
    if abs(x) > thresh_error:
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
	stim_oxy=nengo.Node(output=oxy_function)
	stim_motor=nengo.Node(output=1)

	#ENSEMBLES ########################################################################

	#stimulus subpopulations
	U=nengo.Ensemble(N,dim) #intermediary#
	C=nengo.Ensemble(N,dim) #excited by stim_CS
	Context=nengo.Ensemble(N,dim) #excited by stim_CS
	Motor=nengo.Ensemble(N,dim) #indicates movement or freezing

	#Lateral Amygdala subpopulations
	#lateral amygdala, learns associations b/w CS and US (no extinction)
	LA=nengo.Ensemble(4*N,2*dim,radius=2) 
	#GABA application targets are local GABAergic interneurons in LA which control 
	#excitability-dependent synaptic plasticity, and therefore fear conditioning,
	#as well as control activity of LA, reducing fear response
	#This population has one extra dimension, "i", which is excited by the GABA stimulus
	LA_inter=nengo.Ensemble(8*N,2*dim+1,radius=2,n_eval_points=3000,
	        encoders=Choice([[1,0,0],[0,1,0],[0,0,1]]),
	        eval_points=Uniform(thresh_inter,1))
	        
	#Intercalated Cells
	ITCd=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1)) 
	ITCv=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1))

    #Central Lateral and Central Medial Amygdala subpopulations
	CeL_ON=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeM_DAG=nengo.Ensemble(N,dim) #medial central amygdala, outputs fear responses

	#intra-BA/Cortex/Hippocampus subpopulations
	BA_fear=nengo.Ensemble(N,dim) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(N,dim) #basolateral amygdala cells activated by extinction
	#BA_inter represent several populations whose exact connections are unknown, and may exist
	#within BA or in nearby hippocampus/cortex. The functions of this population are:
	#(a) sustain activity of BA_fear and BA_extinct to produce elongated behavior (integrator->long freeze)
	#(b) mutually inhibit BA_fear and BA_extinct (can't do both at once)
	#(c) provide learning signal for context to BA_fear/BA_extinct populations
	#(d) represent GABAergic activation to allow drug control of (a-c)
	#representation: [context,US,inhibit,Fear_recurrent,Extinct_recurrent]
	BA_inter=nengo.Ensemble(10*N,2*dim+3,radius=3,
	       # encoders=Choice([[1,0,0],[0,1,0],[0,0,1]]),
            intercepts=Exponential(scale=(1 - thresh_inter) / 5.0, shift=thresh_inter, high=1),
            eval_points=Uniform(thresh_inter, 1.1),n_eval_points=5000)
	
	#Error populations
	error_cond=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	error_context=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	error_extinct=nengo.Ensemble(N,dim,encoders=Choice([[1]]), eval_points=Uniform(0, 1))

	#CONNECTIONS ########################################################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	nengo.Connection(stim_gaba,LA_inter[-1],synapse=tau_stim) #stimulate the 'control' dimension
	nengo.Connection(stim_oxy,CeL_OFF,synapse=tau_stim)
	
	#Lateral Amygdala connections
	conn_condition=nengo.Connection(C,LA[:dim],synapse=tau_learn,transform=0)
	nengo.Connection(U,LA[dim:2*dim],synapse=tau) #error signal computed in LA, so it needs US info
	nengo.Connection(LA,LA_inter[:2*dim],synapse=tau_recurrent) #recurrent connection to interneurons
	nengo.Connection(LA_inter,error_cond,synapse=tau_recurrent,function=LA_inter_error)
	nengo.Connection(LA_inter,LA,synapse=tau_recurrent,function=LA_recurrent_out) #recurrent connection to interneurons
            
    #Basal Nuclei connections, includes possible Cortex/Hippocampus connections
	nengo.Connection(LA[:dim],BA_fear,synapse=tau) #CS-fear circuit
	conn_context=nengo.Connection(Context,BA_fear,synapse=tau,transform=0) #context-fear circuit
	conn_extinct=nengo.Connection(Context,BA_extinct,synapse=tau,transform=0) #context-extinction circuit
	nengo.Connection(Context,BA_inter[:dim],synapse=tau) #context for learning connections
	nengo.Connection(U,BA_inter[dim:2*dim],synapse=tau_stim) #US for learning connections
	nengo.Connection(stim_gaba,BA_inter[-3],synapse=tau_stim) #inhibition for gaba control of learn, recurrent
	nengo.Connection(BA_fear,BA_inter[-2],synapse=tau) #corresponds to known LA to CCK connection
	nengo.Connection(BA_extinct,BA_inter[-1],synapse=tau) #unknown
	nengo.Connection(BA_inter[-2],BA_fear,synapse=tau_recurrent,transform=BA_inter_feedback_excite) #IL
	nengo.Connection(BA_inter[-1],BA_extinct,synapse=tau_recurrent,transform=BA_inter_feedback_excite) #dne?
	nengo.Connection(BA_inter[-2],BA_extinct,synapse=tau_recurrent,transform=BA_inter_feedback_inhibit) #CCK
	nengo.Connection(BA_inter[-1],BA_fear,synapse=tau_recurrent,transform=BA_inter_feedback_inhibit) #PV
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
	nengo.Connection(error_cond,conn_condition.learning_rule,synapse=tau_learn,function=error_out)
	conn_context.learning_rule_type=nengo.PES(learning_rate=context_PES_rate)
	nengo.Connection(error_context,conn_context.learning_rule,synapse=tau_learn,function=error_out)
	conn_extinct.learning_rule_type=nengo.PES(learning_rate=extinct_PES_rate)
	nengo.Connection(error_extinct,conn_extinct.learning_rule,synapse=tau_learn,function=error_out)
	
	#Motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=tau) #high=movement
	
	
	#PROBES ########################################################################

	motor_probe=nengo.Probe(Motor,synapse=0.01,sample_every=dt_sample)

'''simulation ###############################################'''

columns=('motor','trial','time','drug')
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
		for t in timesteps:
			motor=sim.data[motor_probe][t][0]
			realtime=(t*dt_sample-t_train)*60 #starts at 0 when training ends, units=realtime seconds
			dataframe.loc[i]=[motor,n,realtime,drug]
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
	sns.barplot(x="drug",y="motor",hue='drug',data=dataframe,ax=ax1)
	sns.tsplot(time="time", value="motor",
					unit="trial", condition="drug",
					data=dataframe,ax=ax2)
	ax2.set(xlabel='time (s)', ylabel='mean(motor)')
else:
	figure, ax1 = plt.subplots(1, 1)
	sns.set(context='paper')
	sns.tsplot(time="time", value="motor",
					unit="trial", condition="drug",
					data=dataframe,ax=ax1)
	ax1.set(xlabel='time (s)', ylabel='mean(motor)')
figure.savefig(fname+'.png')
plt.show()