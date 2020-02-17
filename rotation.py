# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 6 of Sanders et al 2020
# simulations of cue rotation experiments
# fig1 is panel B in the paper
# fig2 is panel C in the paper
"""

from model import Animal, WorldModel, CRP, Gauss, VonMises, GenericMixed

import numpy as np
from scipy import stats
from scipy.misc import logsumexp


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
alpha = params['alpha']
sd=np.pi*params['sigma']


#%%

worldModel = WorldModel(CRP(alpha), VonMises())

def experiment(nSpuriousDims=0, rotation=0):
    np.random.seed(params['seed'])
    animal1 = Animal(worldModel=worldModel)
    
    eshape = (1,1+nSpuriousDims)
    #base_experience = np.random.random(eshape)*2*np.pi - np.pi
    base_experience = np.zeros(eshape)
    e = np.empty(eshape)
    for i in range(10):
        sd1 = sd
        sd2 = sd*3
        e[0,0] = base_experience[0,0] + stats.norm.rvs(0,sd1)
        e[0,1:] = base_experience[0,1:] + stats.norm.rvs(0, sd2, size=(1,nSpuriousDims))
        animal1.experience(e, calculate_belief=False)
    e = base_experience
    e[0,0] += rotation
    b, orientations = animal1.experience(e, calculate_belief=True, orient=True, normalize=False)
    print()
    print('nSpuriousDims', nSpuriousDims, 'rotation', rotation)
    print(b)
    print(orientations, orientations*180/np.pi)
    return b, orientations


#%%
nExperiments = 3
dirtyDims = 5

all_post = np.empty((nExperiments,2))
all_orientations = np.empty((nExperiments,2))
b, orientations = experiment(rotation=np.pi)
all_post[0,:] = b.transpose()
all_orientations[0,:] = orientations.transpose()
b, orientations = experiment(nSpuriousDims=dirtyDims, rotation=np.pi)
all_post[1,:] = b.transpose()
all_orientations[1,:] = orientations.transpose()
b, orientations = experiment(nSpuriousDims=dirtyDims, rotation=np.pi/4)
all_post[2,:] = b.transpose()
all_orientations[2,:] = orientations.transpose()

#%%
figInd += 1
f=plt.figure(figInd)
f.clear()

bar_width=0.2
for i in range(nExperiments):
    plt.bar(np.array([0])+i*bar_width, all_post[i,0]-all_post[i,1], bar_width)
plt.xticks(np.array([0, bar_width, 2*bar_width]), ['180$\degree$, clean','180$\degree$, dirty', '45$\degree$, dirty'])
adjustPlot(plt.gca(), fuzzyzero=True)
plt.ylabel('State Evidence Ratio')
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

#%%
figInd += 1
f=plt.figure(figInd)
f.clear()
ax = plt.subplot(111, projection='polar')
for i in range(nExperiments):
    c = ax.bar(all_orientations[i,0], 1, width=0.1)
ax.set_rticks([])
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

