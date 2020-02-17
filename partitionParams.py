#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 3 and 4 of Sanders et al 2020
# performs inference over different trainings
# figs1&3 are for Fig. 3 from the paper
# figs4&5 are for Fig. 4 from the paper
"""
from model import Animal, WorldModel, CRP, Gauss

import numpy as np
from scipy import stats
from scipy.special import logsumexp


from myutils import stripFilename, binstarts2binedges, adjustPlot, params
from matplotlib import pyplot as plt
plt.style.use('ggplot')
try:
    filename = stripFilename(__file__)
except NameError:
    filename = 'script'
figInd = 0


#%% parameter setup

alpha_default = params['alpha']     #1e-3
sd_default = 0.3 #params['sigma']         # std dev of features

sd0_default = 1        # prior std dev
nPresentations = 1      # # of presentations per experience
N_default = int(50/nPresentations)      # max # of experiences
distance_default = 2    # distance between different features
K = 1           # # of features

#%%
def partitions2activity(animal, experience, partitionsList, patternsList):
    # given a single experience, an animal with its history, 
    # and sets of partitions to consider along with their associated activity patterns,
    # outputs an activity pattern, along with the underlying posteriors of each partition and beleifs
    # partitionsList is a list of lists.  The outer list is a list of partitions to consider;
    # the inner lists are lists of cluster assignments for each past experience the animal has had
    # patternsList is a list of 1D activity arrays.  
    # the number of arrays = the total number of clusters in all partitions (no patterns for novel clusters necessary)
    #
    # lps is length # of partitions
    # belief state is length sum(# of clusters + 1 for each partition)
    nPartitions = len(partitionsList)
    nPossibilities = sum([np.max(partitionsList[i])+2 for i in range(nPartitions)])
    _nN = len(patternsList[0])
    patternsArray = np.nan*np.empty((nPossibilities, _nN))
    lps = np.nan*np.empty((nPartitions, ))
    beliefState = np.nan*np.empty((1, nPossibilities))
    n0=0    # for indexing patternsList
    n1=0    # for indexing patternsArray (includes novel clusters)
    for i, partition in enumerate(partitionsList):
        lps[i] = animal.partition_prob(partition)
        post = animal.belief(experience, partition)
        C = np.max(partition)+1       # # of clusters in that partition
        for j in range(C):
            beliefState[0,n1] = lps[i] + post[j]
            patternsArray[n1,:] = patternsList[n0]
            n0+=1
            n1+=1
        beliefState[0,n1] = lps[i] + post[C]
        patternsArray[n1,:] = np.random.rand(_nN)    # novel pattern for novel cluster
        n1+=1
    beliefState -= logsumexp(lps)   # lps hasn't been normalized, but each post has been
    activity = np.dot(np.exp(beliefState), patternsArray)
    return activity, lps, beliefState

        
            
            
    
    


#%% run experiment
nN = 100    # # of neurons
zeta0 = np.random.rand(nN)      # cluster together
zeta1A = np.random.rand(nN)     # cluster separately, cluster A
zeta1B = np.random.rand(nN)     # cluster separately, cluster B
def experiment(alpha=alpha_default, sd0=sd0_default, sd=sd_default, N=N_default, distance=distance_default):
    np.random.seed(1)
    _model = WorldModel(CRP(alpha), Gauss(sigma_0=sd0))
    _animal = Animal(worldModel=_model)
    
    
    experiences = np.nan*np.ones((N*nPresentations*2))
    partition1 = np.array([])
    partition2 = np.array([])
    partitionN = np.array([])
    lps = np.nan*np.empty((N, 3))
    posts = np.nan*np.empty((N*nPresentations*2, 3))    
    similarity = np.nan*np.empty((N, 1))
    for i in range(N):      
        c1 = stats.norm(loc=np.array([-distance/2]), scale=sd*np.ones((1,K)))
        c2 = stats.norm(loc=np.array([+distance/2]), scale=sd*np.ones((1,K)))
        
        for _i in range(nPresentations):
            experience1 = c1.rvs()
            experiences[i*nPresentations*2+_i*2] = experience1
            post1 = _animal.experience(experience1)
            if i + _i == 0:
                partition1 = np.array([[0]])
                partition2 = np.array([[0]])
                partitionN = np.array([[0]])
            else:
                partition1 = np.vstack((partition1, 0))
                partition2 = np.vstack((partition2, 0))
                partitionN = np.vstack((partitionN, 2*i*nPresentations+_i))
            if i>0:
                posts[i*nPresentations*2+_i,:] = post1.squeeze()
                _animal.clusterAssignments = partition2
        for _i in range(nPresentations):
            experience2 = c2.rvs()
            experiences[i*nPresentations*2+_i*2+1] = experience2
            post2 = _animal.experience(experience2)
            partition1 = np.vstack((partition1, 0))
            partition2 = np.vstack((partition2, 1))
            partitionN = np.vstack((partitionN, (2*i+1)*nPresentations+_i))
            if i>0:
                posts[i*nPresentations*2+nPresentations+_i,:] = post2.squeeze()
            _animal.clusterAssignments = partition2
         
        #TODO: look over this for what to do about partitions2activity
        ret = partitions2activity(_animal, experience1, [partition1, partition2], [zeta0, zeta1A, zeta1B])
        activity1 = ret[0]
        tmp_lps = ret[1]
        ret = partitions2activity(_animal, experience2, [partition1, partition2], [zeta0, zeta1A, zeta1B])   
        activity2 = ret[0]
        tmp_lps_2 = ret[1]
        np.testing.assert_array_equal(tmp_lps, tmp_lps_2)
        lps[i,0:2] = tmp_lps
        lps[i, 0] = _animal.partition_prob(partition1)
        lps[i, 1] = _animal.partition_prob(partition2)
        lps[i, 2] = _animal.partition_prob(partitionN)
        similarity[i] = np.corrcoef(activity1, activity2)[0,1]
    return similarity, lps, _animal, partition1, partition2, c1, c2, experiences, posts
   



#%%
if __name__ == '__main__':
    figInd=0
    lw=0.7
    N=15
    rets = experiment(N=N)
    similarity = rets[0]
    lps = rets[1]
    experiences = rets[7]
    posts = rets[8]
    
    figInd += 1
    plt.figure(figInd)
    plt.clf()
    plt.plot(np.arange(N)+1,-(lps[:,0]-lps[:,1]), 'g')
    plt.ylim((-50/2,50/2))
    adjustPlot(plt.gca(), fuzzyzero=True)
    plt.xlabel('# of experiences')
    plt.ylabel('Negative Partition Evidence Ratio')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

    figInd += 1
    plt.figure(figInd)
    plt.clf()
    plt.subplot(2,2,1)
    plt.plot(np.arange(0,N,0.5)+1,np.max(posts[:,0:2], axis=1)-posts[:,2])
    plt.ylim([-20,20])
    adjustPlot(plt.gca(), fuzzyzero=True)
    plt.xlabel('# of experiences')
    plt.ylabel('State Evidence Ratio')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
    
    
    
    rets = experiment(N=10)
    experiences = rets[7]
    animal = rets[2]
    partition1 = rets[3]
    partition2 = rets[4]
    
    experiences1 = experiences[np.squeeze(partition1==0)]
    experiences2a = experiences[np.squeeze(partition2==0)]
    experiences2b = experiences[np.squeeze(partition2==1)]
    experiences1 = np.reshape(experiences1, (len(experiences1),-1))
    experiences2a = np.reshape(experiences2a, (len(experiences2a),-1))
    experiences2b = np.reshape(experiences2b, (len(experiences2b),-1))
    _, dist1 = animal.worldModel.observationModel.posteriorPredictive(experiences[0], experiences1, return_dist=True)
    _, dist2a = animal.worldModel.observationModel.posteriorPredictive(experiences[0], experiences2a, return_dist=True)
    _, dist2b = animal.worldModel.observationModel.posteriorPredictive(experiences[0], experiences2b, return_dist=True)
    x= np.linspace(-distance_default,distance_default,5000)
    figInd += 1
    plt.figure(figInd)
    plt.clf()
    plt.plot(x, dist1.pdf(x.reshape(-1,1)), '--r', linewidth=lw, label='1 State')
    plt.plot(x, dist2b.pdf(x.reshape(-1,1))/2, '--b', linewidth=lw, label='2 States')
    plt.plot(x, dist2a.pdf(x.reshape(-1,1))/2, '--b', linewidth=lw, label=None)
    plt.plot(experiences, 0*np.array(experiences), '.k', markersize=9, label='Observations')
    adjustPlot(plt.gca(), fuzzyzero=False)
    plt.xlim([np.min(x), np.max(x)])
    plt.ylabel('Probability Density')
    plt.xlabel('Feature 1')
    plt.legend()
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
    
    
    rets = experiment(N=15,distance=0)
    similarity = rets[0]
    lps = rets[1]
    experiences = rets[7]
    figInd += 1
    plt.figure(figInd)
    plt.clf()
    plt.plot(lps[:,0]-lps[:,1], 'g')
    plt.ylim((-50/2,50/2))
    adjustPlot(plt.gca(), fuzzyzero=True)
    plt.xlabel('# of experiences')
    plt.ylabel('Partition Evidence Ratio')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

    rets = experiment(N=10, distance=0)
    experiences = rets[7]
    animal = rets[2]
    partition1 = rets[3]
    partition2 = rets[4]
    
    experiences1 = experiences[np.squeeze(partition1==0)]
    experiences2a = experiences[np.squeeze(partition2==0)]
    experiences2b = experiences[np.squeeze(partition2==1)]
    experiences1 = np.reshape(experiences1, (len(experiences1),-1))
    experiences2a = np.reshape(experiences2a, (len(experiences2a),-1))
    experiences2b = np.reshape(experiences2b, (len(experiences2b),-1))
    _, dist1 = animal.worldModel.observationModel.posteriorPredictive(experiences[0], experiences1, return_dist=True)
    _, dist2a = animal.worldModel.observationModel.posteriorPredictive(experiences[0], experiences2a, return_dist=True)
    _, dist2b = animal.worldModel.observationModel.posteriorPredictive(experiences[0], experiences2b, return_dist=True)
    x= np.linspace(-distance_default,distance_default,5000)
    figInd += 1
    plt.figure(figInd)
    plt.clf()
    plt.plot(x, dist1.pdf(x.reshape(-1,1)), '--r', linewidth=lw, label='1 State')
    plt.plot(x, dist2b.pdf(x.reshape(-1,1))/2, '--b', linewidth=lw, label='2 States')
    plt.plot(x, dist2a.pdf(x.reshape(-1,1))/2, '--b', linewidth=lw, label=None)
    plt.plot(experiences, 0*np.array(experiences), '.k', markersize=9, label='Observations')
    adjustPlot(plt.gca(), fuzzyzero=False)
    plt.xlim([np.min(x), np.max(x)])
    plt.ylabel('Probability Density')
    plt.xlabel('Feature 1')
    plt.legend()
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
    
    
    #%%
    plotParameterSweeps = False
    
    if plotParameterSweeps:
    
        # decent colormaps:
        # YlGn
        # YlGnBu
        # hot, gist_heat
        # viridis
        
        
        numAlphas = 11
        alphas = np.logspace(-15, 5, numAlphas, base=10)
        similarities = np.empty((numAlphas, N_default))
        for j, alpha in enumerate(alphas):
            rets = experiment(alpha)
            s = rets[0]
            similarities[j,:] = np.reshape(s, (1,N_default))
            
        cmap = plt.get_cmap('YlGnBu')
        aPlotVec = binstarts2binedges(np.log10(alphas))
        sPlotVec = binstarts2binedges(np.arange(N_default))
        extent = sPlotVec[0], sPlotVec[-1], aPlotVec[-1], aPlotVec[0]
        
        figInd += 1
        plt.figure(figInd)
        plt.clf()
        plt.imshow(similarities, vmin=0, vmax=1, extent=extent, cmap=cmap, aspect='auto')
        plt.ylabel('Log10(alpha)')
        plt.xlabel('# of experiences')
        plt.colorbar()
        plt.title('Similarity betweeen environments')
        plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
        
        
        #%%
        numSd0s = 11
        sd0s = np.logspace(-2, 3, numSd0s, base=10)
        similarities = np.empty((numSd0s, N_default))
        for j, sd0 in enumerate(sd0s):
            rets = experiment(sd0=sd0)
            s = rets[0]
            similarities[j,:] = np.reshape(s, (1,N_default))
            
        cmap = plt.get_cmap('YlGnBu')
        aPlotVec = binstarts2binedges(np.log10(sd0s))
        sPlotVec = binstarts2binedges(np.arange(N_default))
        extent = sPlotVec[0], sPlotVec[-1], aPlotVec[-1], aPlotVec[0]
        
        figInd += 1
        plt.figure(figInd)
        plt.clf()
        plt.imshow(similarities, vmin=0, vmax=1, extent=extent, cmap=cmap, aspect='auto')
        plt.ylabel('Log10(Prior SD)')
        plt.xlabel('# of experiences')
        plt.colorbar()
        plt.title('Similarity betweeen environment representations')
        plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
        
        
        #%%
        numSds = 6
        sds = np.linspace(0, 0.5, numSds)
        similarities = np.empty((numSds, N_default))
        for j, sd in enumerate(sds):
            rets = experiment(sd=sd)
            s = rets[0]    
            similarities[j,:] = np.reshape(s, (1,N_default))
            
        cmap = plt.get_cmap('YlGnBu')
        aPlotVec = binstarts2binedges(sds)
        sPlotVec = binstarts2binedges(np.arange(N_default))
        extent = sPlotVec[0], sPlotVec[-1], aPlotVec[-1], aPlotVec[0]
        
        figInd += 1
        plt.figure(figInd)
        plt.clf()
        plt.imshow(similarities, vmin=0, vmax=1, extent=extent, cmap=cmap, aspect='auto')
        plt.ylabel('Generative SD')
        plt.xlabel('# of experiences')
        plt.colorbar()
        plt.title('Similarity betweeen environment representations\nalpha=' + str(alpha_default))
        plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
