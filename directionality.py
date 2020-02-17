# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 5 of Sanders et al 2020
# compares directed vs random circular observations
# fig1 is scatter plot of circular experiences in the two training paradigms
# fig2 is bar graph of partition evidence ratios in both cases
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

#%% functions
mu=0
sd=np.pi*params['sigma'] #0.1
np.random.seed(params['seed'])


class Directional:
    # This distribution generates a bimodal distribution in one dimension and 
    # a unimodal distrbution in nSpuriousDims dimensions
    def __init__(self, nSpuriousDims=0):
        self.nSpuriousDims = nSpuriousDims
        if nSpuriousDims >0:
            mus1 = 0*np.ones((1,nSpuriousDims+1))
            mus2 = 0*np.ones((1,nSpuriousDims+1))
            mus1[0,0] = mu
            mus2[0,0] = mu+np.pi
            sds = sd*np.ones((1,nSpuriousDims+1))
            
            self.dist1 = stats.norm(loc=mus1, scale=sds)
            self.dist2 = stats.norm(loc=mus2, scale=sds)
        else:
            self.dist1 = stats.vonmises(kappa=10, loc=0)
            self.dist2 = stats.vonmises(kappa=10, loc=np.pi)           
        self.last = 2
    
    def rvs(self):
        if self.last == 1:
            self.last = 2
            rv = self.dist2.rvs()
            if self.nSpuriousDims > 0:
                rv[0] = np.mod(rv[0], 2*np.pi)
            return rv.reshape(1,-1) # change vector to single-row array
        else:  #if self.last == 1:
            self.last = 1
            rv = self.dist1.rvs()
            if self.nSpuriousDims > 0:
                rv[0] = np.mod(rv[0], 2*np.pi)
            return rv.reshape(1,-1) # change vector to single-row array
        

class NonDirectional:
    # This distribution generates a circular uniform distribution in one dimension and 
    # a unimodal distrbution in nSpuriousDims dimensions    
    def __init__(self, nSpuriousDims=0):
        self.nSpuriousDims = nSpuriousDims
        mus = 0*np.ones((1,nSpuriousDims))
        sds = sd*np.ones((1,nSpuriousDims))
        self.spuriousDist = stats.norm(loc=mus, scale=sds)
        
    def rvs(self):
        if self.nSpuriousDims==0:
            return stats.uniform(loc=0, scale=2*np.pi).rvs()
        else:
            a = stats.uniform(loc=0, scale=2*np.pi).rvs()
            rest = self.spuriousDist.rvs()
            ret = np.hstack((a,rest))
            if len(ret.shape) == 1:
                ret = ret.reshape(1,-1)
            return ret


       
def findMidpoint(experiences):
    # input is 1D array where only the directional component of the experience is included.
    directions = np.sort(experiences)
    i = 0
    nlefts = np.nan*np.ones((100*2,))
    distances = np.nan*np.ones((100*2,))
    thetas = np.arange(0, 2*np.pi, np.pi/100)
    for theta in thetas:
        # count number of observations between [theta, theta+pi] 
        nlefts[i] = np.count_nonzero(np.mod(directions - theta, 2*np.pi)<np.pi)
        # calculate distance from theta to all observations and map those to [-pi/2, pi/2]
        # then finds the minimum squared distance
        distances[i] = np.min(np.square(np.mod(experiences- theta +np.pi/2, np.pi)-np.pi/2))
        i+=1
    mask = (nlefts == nlefts[np.argmin(np.abs(nlefts-len(experiences)/2))])
    ind = np.argmax(distances)#*mask)
    return thetas[ind]


#%% simulation
nSpuriousDims = 0
c1 = Directional(nSpuriousDims=nSpuriousDims)
c2 = NonDirectional(nSpuriousDims=nSpuriousDims)

nExperiences = 10
alpha = params['alpha']

distList = [VonMises()]
for obj in range(nSpuriousDims):
    distList.append(Gauss())
worldModel = WorldModel(CRP(alpha), GenericMixed(distList))
animal1 = Animal(worldModel=worldModel)
animal2 = Animal(worldModel=worldModel)

for i in range(nExperiences):
    e = c1.rvs()
    animal1.experience(e, calculate_belief=False)

    e = c2.rvs()
    animal2.experience(e, calculate_belief=False)

#%% plot experiences
figInd = 0
figInd += 1
f=plt.figure(figInd)
f.clear()
ax = plt.subplot(111, projection='polar')
c = ax.scatter(animal1.experiences[:,0], 0.9*np.ones(animal1.experiences.shape[0]), c='b')
c = ax.scatter(animal2.experiences[:,0], 1.1*np.ones(animal1.experiences.shape[0]), c='r')
midpoint1 = findMidpoint(animal1.experiences[:,0])
plt.plot([midpoint1, midpoint1+np.pi], [2,2], c='b')
midpoint2 = findMidpoint(animal2.experiences[:,0])
plt.plot([midpoint2, midpoint2+np.pi], [2,2], c='r')
plt.yticks([])
plt.legend(('Directed', 'Random'), loc='upper right')
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')


#%% calculate log posteriors of 1 vs 2 clusters based on midpoint
# list partitions, vectors of t/f
midpoint1 = findMidpoint(animal1.experiences[:,0])
midpoint2 = findMidpoint(animal2.experiences[:,0])
partition1_1 = np.zeros((nExperiences,))    # 1 cluster, animal1
partition1_2 = (np.mod(animal1.experiences[:,0] - midpoint1, 2*np.pi) > np.pi).astype(int)    # 2 clusters, animal1

partition2_1 = np.zeros((nExperiences,))    # 1 cluster, animal2
partition2_2 = np.mod(animal2.experiences[:,0] - midpoint2, 2*np.pi) > np.pi    # 2 clusters, animal2

#%%
# partition_prob
figInd += 1
figInd=2
fig = plt.figure(figInd)
fig.clear()
ax = plt.subplot(1,2,1)
bar_width=0.2
inds = np.array([0,1])-bar_width/2
rat1 = animal1.partition_prob(partition1_1) - animal1.partition_prob(partition1_2)
rat2 = animal1.partition_prob(partition2_1) - animal1.partition_prob(partition2_2)
plt.bar(np.array([-bar_width/2, +bar_width/2]), -np.array([rat2, rat1]), bar_width, color='rb')
m=np.max(np.abs(plt.ylim()))
plt.ylim([-m,m])
adjustPlot(ax, fuzzyzero=True)
plt.xticks(np.array([-bar_width/2, +bar_width/2]), ['Random','Directed'])
plt.ylabel('Negative Partition Evidence Ratio')
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

