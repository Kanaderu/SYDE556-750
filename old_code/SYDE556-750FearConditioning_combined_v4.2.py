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
filename='FearConditioningMullerCombinedV4pt2'
experiment='viviani' #muller-tone, muller-context, viviani
drugs=['none','oxytocin'] #['saline-saline','muscimol-saline','saline-muscimol','muscimol-muscimol']# 
n_trials=2
pairings_train=5 #how many CS-US pairs to train on
tones_test=1
dt=0.001 #timestep
dt_sample=0.01 #probe sample_every
condition_PES_rate = 5e-4 #conditioning learning rate to CS
context_PES_rate = 5e-5 #conditioning learning rate to Context
extinct_PES_rate = 5e-6 #extinction learning rate
gaba_muscimol=2.0 #1.5 -> identical gaba responses, 1.0 -> muscimol-saline = saline-saline
gaba_min=0.2 #minimum amount of inhibition
oxy=1.0 #magnitude of oxytocin stimulus

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
LA_to_BA=1.0
ITCd_to_ITCv=-0.75
ITCv_to_CEM_DAG=-1.0
LA_inter_feedback=-0.8 #controlls recurrent inhibition in LA; negative values for inhibition
BA_fear_recurrent=0.1 #controls recurrent excitation in BA_fear
BA_extinct_recurrent=0.1 #controls recurrent excitation in BA_extinct
CCK_feedback=-0.3 #controls mutual inhibition b/w BA_fear and BA_extinct
PV_feedback=-0.5 #controls mutual inhibition b/w  BA_extinct and BA_fear

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
	'LA_to_BA':LA_to_BA,
    'BA_fear_recurrent':BA_fear_recurrent,
    'BA_extinct_recurrent':BA_extinct_recurrent,
    'CCK_feedback':CCK_feedback,
    'PV_feedback':PV_feedback,

	'tt':tt,
	'nt':nt,
	'st':st,
	'n2t':n2t,
	'wt':wt,
	't_train':t_train,
	't_test':t_test,
}

'Helper functions and transformations on ensemble connections ########################'''

def parisien_transform(conn, inh_synapse, inh_proportion=0.25): #https://github.com/nengo/nengo/issues/921
    # only works for ens->ens connections
    assert isinstance(conn.pre_obj, nengo.Ensemble)
    assert isinstance(conn.post_obj, nengo.Ensemble)    

    # make sure the pre and post ensembles have seeds so we can guarantee their params
    if conn.pre_obj.seed is None:
        conn.pre_obj.seed = np.random.randint(0x7FFFFFFF)
    if conn.post_obj.seed is None:
        conn.post_obj.seed = np.random.randint(0x7FFFFFFF)

    # compute the encoders, decoders, and tuning curves
    model2 = nengo.Network(add_to_container=False)
    model2.ensembles.append(conn.pre_obj)
    model2.ensembles.append(conn.post_obj)
    model2.connections.append(conn)
    sim = nengo.Simulator(model2)
    enc = sim.data[conn.post_obj].encoders
    dec = sim.data[conn].weights
    pts, act = nengo.utils.ensemble.tuning_curves(conn.pre_obj, sim)

    # compute the original weights
    transform = nengo.utils.builder.full_transform(conn)
    w = np.dot(enc, np.dot(transform, dec))

    # compute the bias function, bias encoders, bias decoders, and bias weights
    total = np.sum(act, axis=1)    
    bias_d = np.ones(conn.pre_obj.n_neurons) / np.max(total)    
    bias_func = total / np.max(total)    
    bias_e = np.max(-w / bias_d, axis=1)
    bias_w = np.outer(bias_e, bias_d)

    # add the new model compontents
    nengo.Connection(conn.pre_obj.neurons, conn.post_obj.neurons,
                    transform=bias_w,
                    synapse=conn.synapse)
    inh = nengo.Ensemble(n_neurons = int(conn.pre_obj.n_neurons*inh_proportion),
                    radius=conn.pre_obj.radius,
                    dimensions = 1,
                    encoders = nengo.dists.Choice([[1]]))
    nengo.Connection(conn.pre_obj, inh, 
                    solver=nengo.solvers.NnlsL2(),
                    transform=1,
                    synapse=inh_synapse,
                    **nengo.utils.connection.target_function(pts, bias_func))
    nengo.Connection(inh, conn.post_obj.neurons,
                    solver=nengo.solvers.NnlsL2(),
                    transform=-bias_e[:,None])

    return inh #return the inhibitory ensemble for assignment in model

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
    return 0.1 #must be nonzero to allow extinction training

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
	LA=nengo.Ensemble(8*N,dim,radius=2) 
	#GABA application targets are local GABAergic interneurons in LA which control 
	#excitability-dependent synaptic plasticity, and therefore fear conditioning,
	#as well as control activity of LA, reducing fear response.
	#send CS representation to LA_inter, to compute feedback
	#Do this using Terry's Parisien Transform function
	LA_recurrent=nengo.Connection(LA,LA,synapse=tau_recurrent,transform=LA_inter_feedback)
	LA_inter=parisien_transform(LA_recurrent, inh_synapse=LA_recurrent.synapse, inh_proportion=0.5)

	#Intercalated Cells
	ITCd=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) 
	ITCv=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1))

    #Central Lateral and Central Medial Amygdala subpopulations
	CeL_ON=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(N,dim) #ON cells in the lateral central amygdala
	CeM_DAG=nengo.Ensemble(N,dim) #medial central amygdala, outputs fear responses

	#intra-BA/Cortex/Hippocampus subpopulations
	BA_fear=nengo.Ensemble(4*N,dim,radius=2,encoders=Choice([[1]]), intercepts=Uniform(0, 1))
	BA_extinct=nengo.Ensemble(4*N,dim,radius=2,encoders=Choice([[1]]), intercepts=Uniform(0, 1))
	#GABA application targets are local GABAergic interneurons in BA which control 
	#excitability-dependent synaptic plasticity, and therefore context conditioning/extinction.
	#These interneurons also control mutual excitability within BA, such that a fear response
	#inhibits and extinction response and visa versa.
	#Do this using Terry's Parisien Transform function
	CCK_conn=nengo.Connection(BA_fear,BA_extinct,synapse=tau_recurrent,transform=CCK_feedback)
	PV_conn=nengo.Connection(BA_extinct,BA_fear,synapse=tau_recurrent,transform=PV_feedback)
	CCK=parisien_transform(CCK_conn, inh_synapse=CCK_conn.synapse)
	PV=parisien_transform(PV_conn, inh_synapse=PV_conn.synapse)
	
	
	#Error populations
	error_cond=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) #US-CS
	error_context=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) #US-F
	error_extinct=nengo.Ensemble(N,dim,encoders=Choice([[1]]), intercepts=Uniform(0, 1)) #E-US

	#CONNECTIONS ########################################################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
	nengo.Connection(stim_gaba,LA_inter,synapse=tau_drug) #stimulate the gabaergic interneurons directly
	nengo.Connection(stim_gaba,CCK,synapse=tau_drug) #activate BA gabaergic interneurons
	nengo.Connection(stim_gaba,PV,synapse=tau_drug) 
	nengo.Connection(stim_oxy,CeL_OFF,synapse=tau_stim) #stimulate the gabaergic interneurons (P trans?)
	        
	#Lateral Amygdala connections
	conn_condition=nengo.Connection(C,LA,synapse=tau_stim,transform=0) #primary learned connection CS-freeze
	nengo.Connection(LA,error_cond,synapse=tau,transform=-1) #send CS response information to error
	nengo.Connection(U,error_cond,synapse=tau_stim) #send US stimulus to error

    #Basal Nuclei connections, includes possible Cortex/Hippocampus connections
	nengo.Connection(LA,BA_fear,synapse=tau,transform=LA_to_BA) #CS-fear circuit
	conn_context=nengo.Connection(Context,BA_fear,synapse=tau_stim,transform=0) #context-fear circuit
	conn_extinct=nengo.Connection(Context,BA_extinct,synapse=tau_stim,transform=0) #context-extinction circuit
	nengo.Connection(BA_fear,BA_fear,synapse=tau_recurrent,transform=BA_fear_recurrent)
	nengo.Connection(BA_extinct,BA_extinct,synapse=tau_recurrent,transform=BA_extinct_recurrent)
	nengo.Connection(BA_fear,error_context,synapse=tau,transform=-1)
	nengo.Connection(BA_fear,error_extinct,synapse=tau)
	nengo.Connection(U,error_context,synapse=tau_stim)
	nengo.Connection(U,error_extinct,synapse=tau_stim,transform=-1)
	
	#Intercalated Cells connections
	nengo.Connection(LA,CeL_ON,synapse=tau)
	nengo.Connection(LA,ITCd,synapse=tau) #CeL pathway
	nengo.Connection(BA_extinct,ITCv,synapse=tau)
	nengo.Connection(ITCd,ITCv,transform=ITCd_to_ITCv,synapse=tau)

	#Central Lateral and Central Medial Amygdala connections
	nengo.Connection(BA_fear,CeM_DAG,synapse=tau)
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=tau)	
	nengo.Connection(ITCv,CeM_DAG,transform=ITCv_to_CEM_DAG,synapse=tau)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=tau)
	nengo.Connection(CeL_ON,CeM_DAG,synapse=tau)
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
		max_motor, min_motor=1.2,-1.2 #np.max(sim.data[motor_probe]),np.min(sim.data[motor_probe])
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