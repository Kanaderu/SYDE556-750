import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

n_trials=3
freezing={}
for exp in ['tone']:
	print exp
	freezing[exp]={}
	print freezing[exp]
	for subj in ['saline-saline', 'muscimol-saline', 'saline-muscimol', 'muscimol-muscimol']:
		freezing[exp][subj]=np.zeros(n_trials)
		for i in range(n_trials):
			freezing[exp][subj][i]=(np.random.uniform(0,1))
			print freezing[exp][subj]
data=pd.DataFrame(freezing)
print data
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
x="exp"
y="freezing"
sns.barplot(x,y,data=freezing['tone'],ax=ax1)