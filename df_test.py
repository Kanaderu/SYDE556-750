import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataframe=pd.read_pickle("FearConditioningMullerCombinedV214246.pkl")
# print pickle

print 'Plotting...'
experiment='batman'
sns.set(context='paper')
if experiment != 'viviani':
	figure, (ax1, ax2) = plt.subplots(2, 1)
	sns.barplot(x="drug",y="freeze",hue='drug',data=dataframe,ax=ax1)
	sns.tsplot(time="time", value="freeze",
					unit="trial", condition="drug",
					data=dataframe,ax=ax2)
else:
	figure, ax1 = plt.subplots(1, 1)
	sns.tsplot(time="time", value="freeze",
					unit="trial", condition="drug",
					data=dataframe,ax=ax1)
plt.show()