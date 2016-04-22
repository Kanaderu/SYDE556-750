# Peter Duggins
# SYDE 556/750
# April 2016
# Final Project - Oxytocin and Fear Conditioning

import nengo
from nengo.dists import Choice,Exponential,Uniform
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
learn_rate = 5e-4 #first order conditioning learning rate
learn_syn=0.01

#stimuli
def US_function(t):
    if 0.9<t<1: return 1
    if 1.9<t<2: return 1
    if 2.9<t<3: return 1
    return 0

def CS_function(t):
    if 0.7<t<1: return 1
    if 1.7<t<2: return 1
    if 2.7<t<3: return 1
    return 0

#model definition
model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ####################################

	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)
	stop_learn = nengo.Node([0])

	#ENSEMBLES ####################################

	#PAG subpopulations
	US=nengo.Ensemble(stim_N,1) #US is scalar valued
	U=nengo.Ensemble(ens_N,ens_dim) #intermediary#
    #difference between US and appropriate resopnse (freezing), 0-1 to prevent extinction learning
	Error=nengo.Ensemble(ens_N, ens_dim, encoders=Choice([[1]]), eval_points=Uniform(0, 1))

	#Amygdala subpopulations
	LA=nengo.Ensemble(ens_N,ens_dim) #lateral amygdala, learns associations
	BA_fear=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala activated by fear
	BA_extinct=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala cells activated by extinction
	BA_int1=nengo.Ensemble(ens_N,ens_dim,
	        encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron
	BA_int2=nengo.Ensemble(ens_N,ens_dim,
	        encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #basolateral amygdala interneuron
	ITCd=nengo.Ensemble(ens_N,ens_dim,
        	encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
	ITCv=nengo.Ensemble(ens_N,ens_dim,
        	encoders=Choice([[1]]), eval_points=Uniform(0, 1)) #intercalated neurons between LA and Ce
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
	nengo.Connection(U,Error,synapse=ens_syn)
	
	#Amygdala connections
	nengo.Connection(LA,BA_fear,synapse=ens_syn) #normal fear circuit
	nengo.Connection(BA_fear,CeM,synapse=ens_syn)
	
	nengo.Connection(LA,ITCd,synapse=ens_syn) #CeL pathway
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(LA,CeL_ON,synapse=ens_syn)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(CeL_OFF,CeM,transform=-1)
	
	nengo.Connection(LA,BA_int1,synapse=ens_syn) #BA pathway
	nengo.Connection(BA_int1,BA_extinct,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_extinct,ITCv,synapse=ens_syn)
	nengo.Connection(ITCv,CeM,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_fear,BA_int1,synapse=ens_syn)
	nengo.Connection(BA_extinct,BA_int2,synapse=ens_syn)
	nengo.Connection(BA_int2,BA_fear,transform=-1,synapse=ens_syn)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=ens_syn)
	
	#context into the BA pathway

	#motor output
	nengo.Connection(CeM,M,transform=-1,synapse=ens_syn)

	#Learned connections and Error calculation
	conditioning = nengo.Connection(C,LA,function=lambda x: [0]*ens_dim,synapse=learn_syn)
	conditioning.learning_rule_type = nengo.PES(learning_rate=learn_rate)
	nengo.Connection(Error, conditioning.learning_rule, transform=-1)
	nengo.Connection(stop_learn, Error.neurons, transform=-10*np.ones((ens_N, ens_dim)))
	nengo.Connection(CeM, Error,transform=-1,synapse=learn_syn)
