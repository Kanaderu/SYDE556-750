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
stim_syn=0.01 #synaptic time constant of stimuli to populations
ens_syn=0.01 #synaptic time constant between ensembles
first_order_rate = 5e-5 #first order conditioning learning rate
learn_syn=0.02

#stimuli
def US_function(t):
    # cycle through the three US
    if 0<t<1: return 0
    if 1<=t<=2: return 1
    if 2<t<3: return 0
    return 0

def CS_function(t):
    # cycle through the three CS
    if 0<t<1: return 0
    if 1<=t<=2: return 1
    if 2<t<3: return 0
    if 3<t<4: return 1
    return 0

#model definition
model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ####################################

	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)

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

	#Cortex subpopulations
	CS=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS through C

	#Hippocampus subpopulations
	C=nengo.Ensemble(stim_N,stim_dim) #intermediary
	
	#Motor population
	M=nengo.Ensemble(stim_N,stim_dim) #indicates movement or freezing

	#CONNECTIONS ####################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,US,synapse=stim_syn)
	nengo.Connection(stim_CS,CS,synapse=stim_syn)
	nengo.Connection(CS,C,synapse=ens_syn)
	nengo.Connection(US,U,synapse=ens_syn)
	nengo.Connection(U,R,synapse=ens_syn)
	
	#Amygdala connections
	nengo.Connection(LA,BA_fear,synapse=ens_syn) #normal fear circuit
	nengo.Connection(BA_fear,CeM,synapse=ens_syn)
	
	nengo.Connection(LA,ITCd,synapse=ens_syn) #CeL pathway
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(LA,CeL_ON,synapse=ens_syn)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(CeL_OFF,CeM,transform=-1)
	
	nengo.Connection(LA,BA_int1,synapse=ens_syn) #BA pathway
	nengo.Connection(BA_int1,BA_extinct,transform=-0.9,synapse=ens_syn)
	nengo.Connection(BA_extinct,ITCv,synapse=ens_syn)
	nengo.Connection(ITCv,CeM,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_fear,BA_int1,synapse=ens_syn)
	nengo.Connection(BA_extinct,BA_int2,synapse=ens_syn)
	nengo.Connection(BA_int2,BA_fear,transform=-0.9,synapse=ens_syn)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=ens_syn)
	
	#context into the BA pathway

	
	nengo.Connection(CeM,M,transform=-1,synapse=ens_syn)

	#Learned connections
	first_order = nengo.Connection(C,LA,synapse=learn_syn,learning_rule_type=nengo.PES())

	#PROBES ####################################
	US_probe=nengo.Probe(US, synapse=0.01)
	CS_probe=nengo.Probe(CS, synapse=0.01)
	CeM_probe = nengo.Probe(CeM, synapse=0.01)