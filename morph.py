# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 7B of Sanders et al 2020
# simulates training with "circle" and "square" environments, 
# and then testing with intermediate "morph" environments
"""


import numpy as np

from myutils import stripFilename, adjustPlot, params
from matplotlib import pyplot as plt
plt.style.use('ggplot')
filename = stripFilename(__file__)
figInd = 0


from partitionParams import partitions2activity, experiment

#%%
nN = 100    # # of neurons
zeta0 = np.random.rand(nN)      # cluster together
zeta1A = np.random.rand(nN)     # cluster separately, cluster A
zeta1B = np.random.rand(nN)     # cluster separately, cluster B

patternList = [zeta0, zeta1A, zeta1B]
# next line is if you want a random subset to be inactive in each pattern
patternList = [pattern*(np.random.rand(nN)>0.5) for pattern in patternList]

#%% Wills
figInd0=1
f=plt.figure(figInd0)
f.clear()
f=plt.figure(figInd0+1)
f.clear()
beliefState = np.nan*np.empty((3, 11, 5))
nTrainings = [5,25]
for j, nTraining in enumerate(nTrainings):
    rets = experiment(N=nTraining, alpha=params['alpha'])
    similarity, lps, animal, partition1, partition2, c1, c2, _, _ = rets
    
    morphExperiences = np.linspace(c1.moment(1), c2.moment(1), num=11)
    
    activities = np.nan*np.empty((11, nN))
    for i, experience in enumerate(morphExperiences):
        ret = partitions2activity(animal, experience, [partition1, partition2], patternList)   
        activities[i,:] = ret[0]
        beliefState[j,i,:] = ret[2].squeeze()
            
        
    #%%        
    if j>=0:
        f=plt.figure(figInd0)
        plt.gca().set_ylim(auto=True)
        plt.plot(morphExperiences, 
                 np.logaddexp(beliefState[j,:,0], beliefState[j,:,2]) - 
                     np.logaddexp(np.logaddexp(beliefState[j,:,1], beliefState[j,:,3]), beliefState[j,:,4]),
                 label='After ' + str(nTraining) + ' Observations')
        adjustPlot(plt.gca(), fuzzyzero=True)
        plt.legend()
        plt.ylabel('Evidence Ratio\nSame as Square:Different from Square')
        plt.xticks(np.array([-0.5, 0.5]), ['Square','Circle'])
        plt.savefig('figures/' + filename + '-' + str(figInd0) + '.pdf', bbox_inches='tight')      
        
    # learn from morph experiences    
    for i, experience in enumerate(morphExperiences):
        ret = partitions2activity(animal, experience, [partition1, partition2], patternList)   
        activities[i,:] = ret[0]
        beliefState[j,i,:] = ret[2].squeeze()

