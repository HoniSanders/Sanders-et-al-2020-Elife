#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:54:35 2019

@author: honi
"""

import numpy as np
from model import Gauss, VonMises, GenericMixed, WorldModel, Animal, CRP
from scipy import stats


from myutils import stripFilename, binstarts2binedges
from matplotlib import pyplot as plt
plt.style.use('ggplot')
try:
    filename = stripFilename(__file__)
except NameError:
    filename = 'script'
figInd = 0

#%%

# include timestamp feature
# posterior decreases over time
# pre-training changes speed of decrease

class TimestampContext:
    def __init__(self, nSpuriousDims=1):
        mu = 0
        sd = 0.3
        tmp = np.ones((1,nSpuriousDims))
        mus = mu*tmp
        sds = sd*tmp        
        self.c = stats.norm(loc=mus, scale=sds)        
        self.timestamp = 0

        
    def rvs(self):
        ret = np.hstack((self.c.rvs(), self.timestamp))
        ret = np.array([self.c.rvs()])
        self.timestamp += 1
        return ret
    
#%%
np.random.seed(0)
c1 = TimestampContext(nSpuriousDims=2)

alpha = 1e-3

nExperiences = 30
worldModel = WorldModel(CRP(alpha), Gauss())
animal0 = Animal(worldModel=worldModel)
for i in range(nExperiences):
    e = c1.rvs()
    animal0.experience(e)


#%% run gibbs sampling
def gibbs(animal):
    nInitiations = 10
    nSamples = 30 # from model.py
    nExp = animal.experiences.shape[0]
    # List of partitions sampled from posterior for animal 1
    cInferredList1 = np.nan*np.ones((nInitiations*nSamples, nExp))
    for i in range(nInitiations):
        print('Initiation ' + str(i))
        initialization = CRP(alpha).rvs(nExp)
        initialization=np.random.randint(nExp, size=nExp)
        #initialization=np.arange(nExp)
        animal.clusterAssignments = initialization
        cInferredList1[i*nSamples:(i+1)*nSamples,:] = animal.reassessAssignments(return_cInferredList=True)
    return cInferredList1

animal1 = animal0.__copy__()
animal1.experiences = animal1.experiences[:10,:]
cInferredList1 = gibbs(animal1)

animal2 = animal0.__copy__()
animal2.experiences = animal2.experiences[:15,:]
cInferredList2 = gibbs(animal2)

animal3 = animal0.__copy__()
animal3.experiences = animal3.experiences[:20,:]
cInferredList3 = gibbs(animal3)


#%% plot similarities based on gibbs sampling

means = np.nan*np.ones((4,))
stds = np.nan*np.ones((4,))
for cInferredList in [cInferredList1, cInferredList2, cInferredList3]:
    figInd += 1
    fig = plt.figure(figInd)
    fig.clear()
    nExp = cInferredList.shape[1]
    sims = np.empty((nExp,nExp))
    
    for col1 in range(nExp):
        for col2 in range(col1):
            sims[col1,col2] = np.mean(cInferredList[:,col1]==cInferredList[:,col2])
            sims[col2,col1] = sims[col1,col2]
        sims[col1,col1] = 1
    vrange = (0, 1)
    cmap = plt.get_cmap('jet')
    plt.imshow(sims, vmin=vrange[0], vmax=vrange[1], cmap=cmap, aspect='auto')
    plt.title('Probability of Being Clustered Together')
    plt.ylabel('Trial Index')
    plt.xlabel('Trial Index')
    plt.show()
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

    #%%
    lags = np.arange(-nExp+1, nExp)
    alignedSims = np.nan*np.ones((nExp, lags.shape[0]))
    for i in range(nExp):
        alignedSims[i,nExp-i-1:2*nExp-i-1] = sims[i,:]
    
    #%%
    figInd += 1
    fig = plt.figure(figInd)
    fig.clear()
    plt.plot(lags, alignedSims.T)
    plt.xlabel('Time Lag')
    plt.ylabel('Probability of being clustered together')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
    
    #%%
    figInd += 1
    fig = plt.figure(figInd)
    fig.clear()
    plt.errorbar(lags, np.nanmean(alignedSims.T, axis=1), yerr=np.nanstd(alignedSims.T, axis=1)/np.sum(~np.isnan(alignedSims.T), axis=1))
    plt.xlabel('Time Lag')
    plt.ylabel('Probability of being clustered together')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
