# Peter Duggins
# SYDE 556/750
# Jan 25, 2015
# Assignment 1

import numpy as np

class ReLUneuron():

	def __init__(self,x_intercept,max_firing_rate):
		self.xintercept=x_intercept
		self.maxrate=max_firing_rate
		self.alpha = (self.maxrate - 0)/(1 - self.xintercept)	#alpha=slope=(y2-y1)/(x2-x1)

	def rate(self,J):
		b=-self.xintercept*self.alpha	
		rate=self.alpha*J+b	
		return np.max(0,rate)

	def Jbias(self):
		return self.xintercept

	def alpha(self):
		return self.xintercept
