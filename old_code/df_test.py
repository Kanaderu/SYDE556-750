import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path='C:\Users\Peter Duggins\Documents\GitHub\SYDE556-750\data'
filename='\FearConditioningCombinedV4pt55424'
params=pd.read_json(path+filename+'_params.json')
dataframe=pd.read_pickle(path+filename+'_data.pkl')

experiment=params['experiment'][0]
# print experiment
# print dataframe

print 'Plotting...'
sns.set(context='paper')
if experiment=='muller-tone' or experiment=='muller-context':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.barplot(x="drug",y="freeze",data=dataframe.query("phase=='test'"),ax=ax1)
	sns.tsplot(time="time", value="freeze", data=dataframe.query("phase=='test'"),
					unit="trial", condition="drug",ax=ax2)
	ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
elif experiment=='viviani':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.barplot(x="phase",y="freeze",hue='drug',data=dataframe,ax=ax1)
	sns.tsplot(time="time", value="freeze",data=dataframe,
					unit="trial", condition="drug",ax=ax2)
	ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
elif experiment=='validate-tone' or experiment=='validate-context':
	figure, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(10, 1, sharex=True)
	#plot stimuli
	sns.tsplot(time="time",value="c",data=dataframe,unit="trial",ax=ax1)
	sns.tsplot(time="time",value="u",data=dataframe,unit="trial",ax=ax2)
	sns.tsplot(time="time",value="context",data=dataframe,unit="trial",ax=ax3)
	#plot LA, BA_fear, BA_extinct
	sns.tsplot(time="time",value="la",data=dataframe,unit="trial",ax=ax4)
	sns.tsplot(time="time",value="ba_fear",data=dataframe,unit="trial",ax=ax5)
	sns.tsplot(time="time",value="ba_extinct",data=dataframe,unit="trial",ax=ax6)
	# #plot error_cond, error_context, error_extinct
	sns.tsplot(time="time",value="error_cond",data=dataframe,unit="trial",ax=ax7)
	sns.tsplot(time="time",value="error_context",data=dataframe,unit="trial",ax=ax8)
	sns.tsplot(time="time",value="error_extinct",data=dataframe,unit="trial",ax=ax9)
	# #plot freeze
	sns.tsplot(time="time", value="freeze",data=dataframe,unit="trial",ax=ax10)
	ax1.set(title=experiment)
plt.show()