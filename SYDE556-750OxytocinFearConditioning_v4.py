# Peter Duggins
# SYDE 556/750
# April 2016
# Final Project - Oxytocin and Fear Conditioning

import nengo
import nengo_gui
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 20

#ensemble parameters
stim_N=50 #neurons for stimulus populations
stim_dim=1 #dimensionality of CS and context
ens_N=50 #neurons for ensembles
ens_dim=1 #dimensions for ensembles
tau_stim=0.01 #synaptic time constant of stimuli to populations
tau_drug=0.1
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA
first_order_rate = 5e-5 #first order conditioning learning rate
learn_syn=0.02

seed=3
rng=np.random.RandomState(seed=seed)
shock_times=np.zeros((40.0/0.001))
shock_indices_1=rng.uniform(0,7.0/0.001,size=7) #ignore spacing constraint
shock_indices_2=rng.uniform(10.0/0.001,17.0/0.001,size=7)
for i in shock_indices_1:
    shock_times[int(i):int(i)+100]=1
for i in shock_indices_2:
    shock_times[int(i):int(i)+100]=1

#stimuli
def US_function(t):
    return shock_times[int(t/0.001)]

def CS_function(t):
    return 1 #always in conditioning box

def oxy_function(t):
    # if 5<t<6: return 1 #oxytocin application phase
    return 0

def GABA_function(t):
    # if 7<t<8: return 1 #muscimol application phase
    return 0
    
#model definition
model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ####################################

	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)
	stim_oxy=nengo.Node(output=oxy_function)
	stim_GABA=nengo.Node(output=GABA_function)
	stim_motor=nengo.Node(output=1)

	#ENSEMBLES ####################################

	#PAG subpopulations
	US=nengo.Ensemble(stim_N,1) #US is scalar valued
	U=nengo.Ensemble(ens_N,1) #intermediary
	R=nengo.Ensemble(ens_N,1) #excited by stim_US through U, recurrent inhibition to dampen

	#Amygdala subpopulations
	LA=nengo.Ensemble(ens_N,ens_dim) #lateral amygdala, learns associations
	BA_fear=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala cells activated by extinction
	BA_int1=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala interneuron
	BA_int2=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala interneuron
	ITCd=nengo.Ensemble(ens_N,ens_dim) #intercalated neurons between LA and Ce
	ITCv=nengo.Ensemble(ens_N,ens_dim) #intercalated neurons between LA and Ce
	CeL_ON=nengo.Ensemble(ens_N,ens_dim) #ON cells in the lateral central amygdala
	CeL_OFF=nengo.Ensemble(ens_N,ens_dim) #ON cells in the lateral central amygdala
	CeM=nengo.Ensemble(ens_N,ens_dim) #medial central amygdala, outputs fear responses
	CeM_DVC=nengo.Ensemble(ens_N,ens_dim) #controlls output to heart

	#Cortex subpopulations
	CS=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS through C

	#Hippocampus subpopulations
	C=nengo.Ensemble(stim_N,stim_dim) #intermediary
	
	#Motor population
	M=nengo.Ensemble(stim_N,stim_dim) #indicates movement or freezing
	Heart=nengo.Ensemble(stim_N,stim_dim) #indicates heart rate

	#CONNECTIONS ####################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,US,synapse=tau_stim)
	nengo.Connection(stim_CS,CS,synapse=tau_stim)
	nengo.Connection(stim_motor,M,synapse=tau_stim) #move by default
	nengo.Connection(stim_motor,Heart,synapse=tau_stim) #heartbeat by default
	nengo.Connection(stim_oxy,CeL_OFF,transform=5,synapse=tau_drug)
	nengo.Connection(stim_GABA,CeM,synapse=tau_drug)
	nengo.Connection(stim_GABA,CeM_DVC,synapse=tau_drug)
	
	#intermediate connections
	nengo.Connection(CS,C,synapse=tau_Glut)
	nengo.Connection(US,U,synapse=tau_Glut)
	nengo.Connection(U,R,synapse=tau_Glut)
	
	#Amygdala connections
	nengo.Connection(LA,BA_fear,synapse=tau_Glut) #normal fear circuit
	nengo.Connection(BA_fear,CeM,synapse=tau_Glut)
	
	nengo.Connection(LA,ITCd,synapse=tau_Glut) #CeL pathway
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=tau_GABA)
	nengo.Connection(LA,CeL_ON,synapse=tau_Glut)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=tau_GABA) #OFF cells inhibit M
	nengo.Connection(CeL_ON,CeM,synapse=tau_GABA) #ON cells excite M and Heart
	nengo.Connection(CeL_ON,CeM_DVC,synapse=tau_GABA)
	nengo.Connection(CeL_OFF,CeM,transform=-2) #2x stronger than BA pathway
	
	nengo.Connection(LA,BA_int1,synapse=tau_Glut) #BA pathway
	nengo.Connection(BA_int1,BA_extinct,transform=-0.9,synapse=tau_GABA)
	nengo.Connection(BA_extinct,ITCv,synapse=tau_Glut)
	nengo.Connection(ITCv,CeM,transform=-1,synapse=tau_GABA)
	nengo.Connection(BA_fear,BA_int1,synapse=tau_Glut)
	nengo.Connection(BA_extinct,BA_int2,synapse=tau_Glut)
	nengo.Connection(BA_int2,BA_fear,transform=-0.9,synapse=tau_GABA)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=tau_GABA)
	
	#context into the BA pathway

	#motor output (negative = freezing)
	nengo.Connection(CeM,M,transform=-1,synapse=tau_GABA)
	nengo.Connection(CeM_DVC,Heart,transform=-1,synapse=tau_GABA)

	#Learned connections
	first_order = nengo.Connection(C,LA,synapse=learn_syn,learning_rule_type=nengo.PES())

	#PROBES ####################################
	US_probe=nengo.Probe(US, synapse=0.01)
	CS_probe=nengo.Probe(CS, synapse=0.01)
	CeM_probe = nengo.Probe(CeM, synapse=0.01)