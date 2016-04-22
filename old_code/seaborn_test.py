import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb

'''Parameters'''
#simulation parameters
filename='FearConditioningMullerCombinedV4pt3'
experiment='viviani' #muller-tone, muller-context, viviani
if experiment == 'viviani':
	drugs=['none','oxytocin']
else:
	drugs=['saline-saline','muscimol-saline','saline-muscimol','muscimol-muscimol']
n_trials=3
pairings_train=1 #how many CS-US pairs to train on
tones_test=1
dt=0.001 #timestep
dt_sample=0.01 #probe sample_every
condition_PES_rate = 5e-4 #conditioning learning rate to CS
context_PES_rate = 5e-5 #context learning rate
extinct_PES_rate = 5e-7 #extinction learning rate
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
LA_to_BA=0.5
ITCd_to_ITCv=-0.25
ITCv_to_CEM_DAG=-1.0
CeL_ON_to_CeM_DAG=1.0
CeL_OFF_to_CeM_DAG=-1.0
BA_fear_to_CeM_DAG=3.0
LA_inter_feedback=-0.5 #controlls recurrent inhibition in LA; negative values for inhibition
BA_fear_recurrent=0.1 #controls recurrent excitation in BA_fear
BA_extinct_recurrent=0.1 #controls recurrent excitation in BA_extinct
CCK_feedback=-0.1 #controls mutual inhibition b/w BA_fear and BA_extinct
PV_feedback=-0.2 #controls mutual inhibition b/w  BA_extinct and BA_fear

#stimuli
tt=10.0/60.0 #tone time
nt=7.0/60.0 #nothing time #in muller paper nt=9.5/60,st=0.5/60,n2t=0
st=2.0/60.0 #shock time
n2t=1.0/60.0 #nothing time
wt=60.0/60.0 #wait/delay time
t_train=int(pairings_train*(wt+tt)/dt)*dt
t_test=t_train*tones_test/pairings_train #multiply by X/pairings for X tone presentations
t_extinct=t_train

columns=('freeze','time','drug','trial','phase')
trials=np.arange(n_trials)
timesteps=np.arange(0,int((t_train+t_test+t_extinct)/dt_sample))
dataframe = pd.DataFrame(index=np.arange(0, len(drugs)*len(trials)*len(timesteps)),
						columns=columns)

i=0
for drug in drugs:
    for n in trials:
		print 'Running experiment \"%s\", drug \"%s\", trial %s...' %(experiment,drug,n+1)
		max_motor, min_motor=0,1
		for t in timesteps:
			if t*dt_sample < t_train: phase='train'
			elif t_train < t*dt_sample < t_train+t_test: phase='test'
			elif t_train+t_test < t*dt_sample: phase='extinct'
			freeze=np.random.rand()
			realtime=t*dt_sample*60 #in seconds from start of simulation
			dataframe.loc[i]=[freeze,realtime,drug,n,phase]
			i+=1
# print dataframe

figure, (ax1, ax2) = plt.subplots(2, 1)
sns.set(context='paper')
# ipdb.set_trace()
sns.barplot(x="phase",y="freeze",hue='drug',data=dataframe,ax=ax1)
sns.tsplot(time="time", value="freeze",data=dataframe,ax=ax2,
				unit="trial", condition="drug")
ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
plt.show()