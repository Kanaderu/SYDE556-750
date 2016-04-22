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
tau_stim=0.01 #synaptic time constant of stimuli to populations
ens_syn=0.01 #synaptic time constant between ensembles
condition_rate = 5e-5 #first order conditioning learning rate
extinction_rate = 1e-7 #extinction learning rate
learn_syn=0.01
tau_drug=0.1
tau_GABA=0.005 #synaptic time constant for GABAergic cells
tau_Glut=0.01 #combination of AMPA and NMDA

#stimuli
t_train=20
t_control=1
t_oxy=1
t_gaba=1
dt=0.001
stim_length=0.1

rng=np.random.RandomState()
US_array=np.zeros((t_train/dt))
US_times=rng.randint(0,7.0/dt,7)
US_times2=rng.randint(10.0/dt,17.0/dt,7)
for i in US_times:
    US_array[i:i+stim_length/dt]=1
for i in US_times2:
    US_array[i:i+stim_length/dt]=1

def US_function(t):
    if t<t_train: return US_array[int(t/dt)]
    return 0

def CS_function(t):
    return 1

def Context_function(t):
    return CS_function(t)
    
def stop_conditioning_function(t): #learning before testing phase
    if 0<t<t_train: return 0
    return 0

def stop_extinction_function(t): #for testing
    return 0
    
def oxy_function(t):
    if t_train+t_control<t<t_train+t_control+t_oxy: return 0.5 #oxytocin application phase
    return 0

def GABA_function(t):
    if t_train+t_control+t_oxy<t<t_train+t_control+t_oxy+t_gaba: return 0 #gaba application phase
    return 0






'''model definition #################################################'''

model=nengo.Network(label='Oxytocin Fear Conditioning')
with model:

	#STIMULI ####################################

	stim_US=nengo.Node(output=US_function)
	stim_CS=nengo.Node(output=CS_function)
	stim_Context=nengo.Node(output=Context_function)
	stim_oxy=nengo.Node(output=oxy_function)
	stim_GABA=nengo.Node(output=GABA_function)
	stim_motor=nengo.Node(output=1)
	stop_conditioning=nengo.Node(output=stop_conditioning_function)
	stop_extinction=nengo.Node(output=stop_extinction_function)

	#ENSEMBLES ####################################

	#PAG subpopulations
	U=nengo.Ensemble(ens_N,ens_dim) #intermediary#
    #difference between US and appropriate resopnse (freezing), 0-1 to prevent extinction learning
	Error_ON=nengo.Ensemble(ens_N, ens_dim, encoders=Choice([[1]]), eval_points=Uniform(0, 1))
	Motor=nengo.Ensemble(stim_N,stim_dim) #indicates movement or freezing
# 	Heart=nengo.Ensemble(stim_N,stim_dim) #indicates heart rate
	
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
	CeM_DAG=nengo.Ensemble(ens_N,ens_dim) #medial central amygdala, outputs fear responses
# 	CeM_DVC=nengo.Ensemble(ens_N,ens_dim) #controlls output to heart

	#Cortex/Thalamus subpopulations
	C=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS

	#Hippocampus subpopulations
	Context=nengo.Ensemble(stim_N,stim_dim) #excited by stim_CS
	Error_OFF=nengo.Ensemble(ens_N, ens_dim, encoders=Choice([[1]]), eval_points=Uniform(0,1)) #no evidence

	#CONNECTIONS ####################################

	#Connections between stimuli and ensembles
	nengo.Connection(stim_US,U,synapse=tau_stim)
	nengo.Connection(stim_CS,C,synapse=tau_stim)
	nengo.Connection(stim_Context,Context,synapse=tau_stim)
	nengo.Connection(stim_motor,Motor,synapse=tau_stim) #move by default
# 	nengo.Connection(stim_motor,Heart,synapse=tau_stim) #heartbeat by default
	nengo.Connection(stim_oxy,CeL_OFF,synapse=tau_drug)
# 	nengo.Connection(stim_GABA,CeL_OFF,synapse=tau_drug)
	nengo.Connection(stim_GABA,CeM_DAG,transform=-1,synapse=tau_drug)
# 	nengo.Connection(stim_GABA,CeM_DVC,transform=-1,synapse=tau_drug)
	
	#Amygdala connections
	nengo.Connection(LA,BA_fear,synapse=ens_syn) #LA pathway: normal fear circuit
	nengo.Connection(BA_fear,CeM_DAG,synapse=ens_syn)
# 	nengo.Connection(BA_fear,CeM_DVC,synapse=ens_syn)
	
	nengo.Connection(LA,ITCd,synapse=ens_syn) #CeL pathway: oxytocin modulated
	nengo.Connection(ITCd,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(LA,CeL_ON,synapse=ens_syn)
	nengo.Connection(CeL_ON,CeL_OFF,transform=-1,synapse=ens_syn)
	nengo.Connection(CeL_ON,CeM_DAG,synapse=tau_GABA)
# 	nengo.Connection(CeL_ON,CeM_DVC,synapse=tau_GABA)
	nengo.Connection(CeL_OFF,CeM_DAG,transform=-1)
	
	nengo.Connection(LA,BA_int1,synapse=ens_syn) #BA pathway: extinction circuit
	nengo.Connection(BA_int1,BA_extinct,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_extinct,ITCv,synapse=ens_syn)
	nengo.Connection(ITCv,CeM_DAG,transform=-1,synapse=ens_syn)
# 	nengo.Connection(ITCv,CeM_DVC,transform=-1,synapse=ens_syn)
	nengo.Connection(BA_fear,BA_int1,synapse=ens_syn)
	nengo.Connection(BA_extinct,BA_int2,synapse=ens_syn)
	nengo.Connection(BA_int2,BA_fear,transform=-1,synapse=ens_syn)
	nengo.Connection(ITCd,ITCv,transform=-1,synapse=ens_syn)
	
	#motor output
	nengo.Connection(CeM_DAG,Motor,transform=-1,synapse=ens_syn)
# 	nengo.Connection(CeM_DVC,Heart,transform=-1,synapse=tau_GABA)
	
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

	#PROBES ####################################
	CeM_DAG_voltage=nengo.Probe(CeM_DAG.neurons,'voltage')
# 	CeM_DVC_voltage=nengo.Probe(CeM_DVC.neurons,'voltage')
	motor_probe=nengo.Probe(Motor,synapse=0.01)
# 	heart_probe=nengo.Probe(Heart,synapse=0.01)






'''simulation and data analysis ###############################################'''

sim=nengo.Simulator(model)
sim.run(t_train+t_control+t_oxy+t_gaba)

motor_value_control=sim.data[motor_probe][t_train/dt:(t_train+t_control)/dt]
motor_value_oxy=sim.data[motor_probe][(t_train+t_control)/dt:(t_train+t_control+t_oxy)/dt]
motor_value_gaba=sim.data[motor_probe][(t_train+t_control+t_oxy)/dt:(t_train+t_control+t_oxy+t_gaba)/dt]
avg_freezing_control=1.0-1.0*np.average(motor_value_control)
std_freezing_control=np.std(motor_value_control)
avg_freezing_oxy=1.0-1.0*np.average(motor_value_oxy)
std_freezing_oxy=np.std(motor_value_oxy)
avg_freezing_extinction=1.0-1.0*np.average(motor_value_gaba)
std_freezing_extinction=np.std(motor_value_gaba)
height_freezing=[avg_freezing_control,avg_freezing_oxy,avg_freezing_extinction]
std_freezing=[std_freezing_control,std_freezing_oxy,std_freezing_extinction]

#Bar Plots
fig=plt.figure(figsize=(16,8))
ax=fig.add_subplot(111)
ax.bar(np.arange(len(height_freezing)),height_freezing,width=1,yerr=std_freezing)
legend=ax.legend(loc='best',shadow=True)
ax.set_xticks([.5,1.5,2.5])
ax.set_xticklabels(('post-training', 'oxytocin', 'post-extinction'))
ax.set_ylabel('Freezing')
plt.show()

# #With heart rate
# heart_value_control=sim.data[heart_probe][t_train/dt:(t_train+t_control)/dt]
# heart_value_oxy=sim.data[heart_probe][(t_train+t_control)/dt:(t_train+t_control+t_oxy)/dt]
# heart_value_gaba=sim.data[heart_probe][(t_train+t_control+t_oxy)/dt:(t_train+t_control+t_oxy+t_gaba)/dt]

# #Activity vs time
# fig=plt.figure(figsize=(16,8))
# ax=fig.add_subplot(211)
# ax.plot(np.arange(0,t_control,dt),-1*motor_value_control,label="Control")
# ax.plot(np.arange(0,t_oxy,dt),-1*motor_value_oxy,label="Oxytocin")
# ax.plot(np.arange(0,t_gaba,dt),-1*motor_value_gaba,label="GABA")
# legend=ax.legend(loc='best',shadow=True)
# plt.title("Freezing")
# ax=fig.add_subplot(212)
# ax.plot(np.arange(0,t_control,dt),heart_value_control,label="Control")
# ax.plot(np.arange(0,t_oxy,dt),heart_value_oxy,label="Oxytocin")
# ax.plot(np.arange(0,t_gaba,dt),heart_value_gaba,label="GABA")
# legend=ax.legend(loc='best',shadow=True)
# ax.set_xlabel('time')
# ax.set_ylabel('Value')
# plt.title("Heart")
# plt.tight_layout()
# plt.show()

# #IPSC != Voltage, so these can't be compared with Figure S3 in Viviani
# CeM_DAG_voltage_control=sim.data[CeM_DAG_voltage][t_train/dt:(t_train+t_control)/dt]
# CeM_DAG_voltage_oxy=sim.data[CeM_DAG_voltage][(t_train+t_control)/dt:(t_train+t_control+t_oxy)/dt]
# CeM_DAG_voltage_gaba=sim.data[CeM_DAG_voltage][(t_train+t_control+t_oxy)/dt:(t_train+t_control+t_oxy+t_gaba)/dt]

# CeM_DVC_voltage_control=sim.data[CeM_DVC_voltage][t_train/dt:(t_train+t_control)/dt]
# CeM_DVC_voltage_oxy=sim.data[CeM_DVC_voltage][(t_train+t_control)/dt:(t_train+t_control+t_oxy)/dt]
# CeM_DVC_voltage_gaba=sim.data[CeM_DVC_voltage][(t_train+t_control+t_oxy)/dt:(t_train+t_control+t_oxy+t_gaba)/dt]

# avg_IPSC_DAG_control=np.average(CeM_DAG_voltage_control)
# std_IPSC_DAG_control=np.std(CeM_DAG_voltage_control)
# avg_IPSC_DAG_oxy=np.average(CeM_DAG_voltage_oxy)
# std_IPSC_DAG_oxy=np.std(CeM_DAG_voltage_oxy)
# avg_IPSC_DAG_gaba=np.average(CeM_DAG_voltage_gaba)
# std_IPSC_DAG_gaba=np.std(CeM_DAG_voltage_gaba)

# height_DAG=[avg_IPSC_DAG_control,avg_IPSC_DAG_oxy,avg_IPSC_DAG_gaba]
# var_DAG=[std_IPSC_DAG_control,std_IPSC_DAG_oxy,std_IPSC_DAG_gaba]

# avg_IPSC_DVC_control=np.average(CeM_DVC_voltage_control)
# std_IPSC_DVC_control=np.std(CeM_DVC_voltage_control)
# avg_IPSC_DVC_oxy=np.average(CeM_DVC_voltage_oxy)
# std_IPSC_DVC_oxy=np.std(CeM_DVC_voltage_oxy)
# avg_IPSC_DVC_gaba=np.average(CeM_DVC_voltage_gaba)
# std_IPSC_DVC_gaba=np.std(CeM_DVC_voltage_gaba)

# height_DVC=[avg_IPSC_DVC_control,avg_IPSC_DVC_oxy,avg_IPSC_DVC_gaba]
# var_DVC=[std_IPSC_DVC_control,std_IPSC_DVC_oxy,std_IPSC_DVC_gaba]

# fig=plt.figure(figsize=(16,8))
# ax=fig.add_subplot(211)
# ax.bar(np.arange(len(height_DAG)),height_DAG,width=1,yerr=var_DAG)
# ax.set_xticks([.5,1.5,2.5])
# ax.set_xticklabels(('control', 'oxytocin', 'GABA'))
# ax.set_ylabel('DAG IPSC (voltage)')
# ax=fig.add_subplot(212)
# ax.bar(np.arange(len(height_DVC)),height_DVC,width=1,yerr=var_DVC)
# ax.set_xticks([.5,1.5,2.5])
# ax.set_xticklabels(('control', 'oxytocin', 'GABA'))
# ax.set_ylabel('DVC IPSC (voltage)')
# plt.show()