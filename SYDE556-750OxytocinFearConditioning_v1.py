# Peter Duggins
# SYDE 556/750
# April 2016
# Final Project - Oxytocin and Fear Conditioning

import nengo
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['font.size'] = 20

#ensemble parameters
stim_N=50 #neurons for stimulus populations
stim_dim=3 #dimensionality of CS and context
ens_N=50 #neurons for ensembles
ens_dim=3 #dimensions for ensembles
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
    if 0<t<1: return [0,0,0]
    if 1<=t<=2: return [1,1,1]
    if 2<t<3: return [0,0,0]
    if 3<t<4: return [1,1,1]
    return [0, 0, 0]

#model definition
model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ####################################

	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)

	#ENSEMBLES ####################################

	#PAG subpopulations
	US=nengo.Ensemble(stim_N,1) #US is scalar valued
	U=nengo.Ensemble(ens_N,ens_dim) #intermediary
	R=nengo.Ensemble(ens_N,ens_dim) #excited by stim_US through U, recurrent inhibition to dampen
	C=nengo.Ensemble(ens_N,ens_dim) #excited by stim_CS

	#Amygdala subpopulations
	LA=nengo.Ensemble(ens_N,ens_dim) #lateral amygdala, learns associations
	BA=nengo.Ensemble(ens_N,ens_dim) #basolateral amygdala, named BL in Carter
	CeM=nengo.Ensemble(ens_N,ens_dim) #medial central amygdala, outputs fear responses

	#Cortex subpopulations
	CS=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS through C

	#Hippocampus subpopulations
	C=nengo.Ensemble(stim_N,stim_dim) #intermediary

	#CONNECTIONS ####################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,US,synapse=stim_syn)
	nengo.Connection(stim_CS,CS,synapse=stim_syn)

	#Feedforward connections between ensembles
	nengo.Connection(CS,C,synapse=ens_syn)
	nengo.Connection(U,R,synapse=ens_syn)
	nengo.Connection(LA,BA,synapse=ens_syn)
	nengo.Connection(BA,CeM,synapse=ens_syn)

	#Learned connections
	first_order = nengo.Connection(C,LA,synapse=learn_syn,learning_rule_type=nengo.PES())

	#PROBES ####################################
	US_probe=nengo.Probe(US, synapse=0.01)
	CS_probe=nengo.Probe(CS, synapse=0.01)
	CeM_probe = nengo.Probe(CeM, synapse=0.01)

#run
sim=nengo.Simulator(model)
sim.run(6.0)

#plot
fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(111)
ax.plot(sim.trange(),sim.data[US_probe].T[0],label='US') #only dimension
ax.plot(sim.trange(),sim.data[CS_probe].T[0],label='CS') #first dimension
ax.plot(sim.trange(),sim.data[CeM_probe].T[0],label='CeM') #first dimension
ax.set_xlabel('time (s)')
legend=ax.legend(loc='best',shadow=True)
plt.show()