import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path='/home/pduggins/SYDE556-750/data/'
filename='FearConditioningMullerCombinedV4pt354264'
params=pd.read_json(path+filename+'_params.json')
dataframe=pd.read_pickle(path+filename+'.pkl')

experiment=params['experiment'][0]
print experiment
print dataframe

print 'Plotting...'
if experiment != 'viviani':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.set(context='paper')
	sns.barplot(x="drug",y="freeze",data=dataframe.query("phase=='train'"),ax=ax1)
	sns.tsplot(time="time", value="freeze", data=dataframe.query("phase=='train'"),
					unit="trial", condition="drug",ax=ax2)
	ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
else:
	figure, ax1 = plt.subplots(1, 1)
	sns.set(context='paper')
	sns.barplot(x="phase",y="freeze",hue='drug',data=dataframe,ax=ax1)
	sns.tsplot(time="time", value="freeze",data=dataframe,
					unit="trial", condition="drug",ax=ax2)
	ax1.set(xlabel='',ylabel='freezing (%)', ylim=(0.0,1.0), title=experiment)
	ax2.set(xlabel='time (s)', ylabel='freezing (%)', ylim=(0.0,1.0))
plt.show()