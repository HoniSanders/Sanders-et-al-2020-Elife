#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 2 of Sanders et al 2020
# compares observations of varying extent of difference from training
"""

import numpy as np
from model import Gauss, VonMises, GenericMixed, WorldModel, Animal, CRP
from scipy import stats


from myutils import stripFilename, binstarts2binedges, adjustPlot, params
from matplotlib import pyplot as plt
plt.style.use('ggplot')
try:
    filename = stripFilename(__file__)
except NameError:
    filename = 'script'
figInd = 0

#%%
np.random.seed(params['seed'])

sd = 0.2
c1 = stats.norm(loc=np.zeros((1,4)), scale=sd*np.ones((1,4)))

alpha = params['alpha']
worldModel = WorldModel(CRP(alpha), Gauss())
animal1 = Animal(worldModel=worldModel)

nExperiences = 20
for i in range(nExperiences):
    animal1.experience(c1.rvs())
    
animal1.clusterAssignments = np.zeros((nExperiences,1))
distance_default = 1    # distance between different features
a=0
b=distance_default
post = np.vstack((animal1.belief(np.array([a,a,a,a])).transpose(),
                  animal1.belief(np.array([b,a,a,a])).transpose(),
                  animal1.belief(np.array([b,a,a,b])).transpose(),
                  animal1.belief(np.array([b,b,b,b])).transpose()
                  ))


#%%
figInd = 0
figInd += 1
f=plt.figure(figInd)
f.clear()

bar_width=0.2
for i in range(4):
    plt.bar(np.array([0])+i*bar_width, post[i,0]-post[i,1], bar_width)
plt.xticks(np.array([0, bar_width, 2*bar_width, 3*bar_width]), ['No Changed Cues','\nCue 1 Changed', 'Cues 1 and 4 Changed', '\nAll Cues Changed'])
adjustPlot(plt.gca(), fuzzyzero=True)
plt.ylabel('State Evidence Ratio')
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

