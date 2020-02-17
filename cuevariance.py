# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 10 of Sanders et al 2020
# compares a probe along a dimension that had either high or low variance during training
# fig1 is scatter plot of experiences in two training paradigms
# fig2 is bar graph of state evidence ratios in both cases
"""

from model import WorldModel, CRP, Gauss
from myutils import stripFilename, binstarts2binedges, adjustPlot, params

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
plt.style.use('ggplot')
try:
    filename = stripFilename(__file__)
except NameError:
    filename = 'script'
figInd = 0

#%%
np.random.seed(params['seed'])


def preTrainingObservations(nSpuriousDims):
    K = 2 + nSpuriousDims     # # of dimensions of each observation
    loSD = 0.1
    hiSD = 2
    mu = np.zeros((K,))
    sigma = loSD*np.ones((K,))
    # hi variance context: hi-var in dimension 0
    mu1 = np.array(mu)
    sigma1 = np.array(sigma)
    mu1[0] = -2.5*hiSD
    sigma1[0] = hiSD
    hiVContext = stats.norm(loc=mu1, scale=sigma1)
    # lo variance context: hi-var in dimenson 1
    mu2 = np.array(mu)
    sigma2 = np.array(sigma)
    sigma2[1] = hiSD
    loVContext = stats.norm(loc=mu2, scale=sigma2)

    nObs = 20   # used to be 100
    hiVObs = hiVContext.rvs(size=((nObs,K)))   
    loVObs = loVContext.rvs(size=((nObs,K)))
    
    return hiVObs, loVObs


def probeObservation(probeDistance, nSpuriousDims):
    oProbe = np.zeros((2+nSpuriousDims,))
    oProbe[0] = probeDistance
    return oProbe


def experience(preTraining, probe, model):
    post = model.belief(probe, preTraining, np.zeros((preTraining.shape[0],1)))
    return post


def experiment(model, hiVObs, loVObs, oProbe):
    hiVCP = experience(hiVObs, oProbe, model)
    loVCP = experience(loVObs, oProbe, model)
    return (hiVCP, loVCP)


def cueVariance(alpha=1e-10, probeDistance=9, nSpuriousDims=0):
    model = WorldModel(CRP(alpha), Gauss())
    hiVObs, loVObs = preTrainingObservations(nSpuriousDims)
    oProbe = probeObservation(probeDistance, nSpuriousDims)

    hiVCP, loVCP = experiment(model, hiVObs, loVObs, oProbe)
    return (hiVCP, loVCP, hiVObs, loVObs, oProbe)
    
    
def main():
    alpha = params['alpha']
    figInd = 0
    
    probeDistance = 1
    nSpuriousDims = 0
    hiVObs, loVObs = preTrainingObservations(nSpuriousDims)
    oProbe = probeObservation(probeDistance, nSpuriousDims)
    
    model = WorldModel(CRP(alpha), Gauss())
    hiVCP = experience(hiVObs, oProbe, model)
    loVCP = experience(loVObs, oProbe, model)

    figInd += 1
    fig = plt.figure(figInd)
    plt.clf()
    h1, = plt.plot(hiVObs[:,0], hiVObs[:,1], '.c')
    h2, = plt.plot(loVObs[:,0], loVObs[:,1], '.m')
    h3, = plt.plot(oProbe[0], oProbe[1], 'xr')
    plt.autoscale(tight=False)
    m = max(plt.axis())
    plt.axis((-m,m,-m,m))
    plt.axis('equal')
    adjustPlot(plt.gca())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend([h1, h2, h3], ['High Variance Training', 'Low Variance Training', 'Probe'], loc='best')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

    figInd += 1
    fig = plt.figure(figInd)
    fig.clear()
    ax = plt.subplot(1,2,1)
    bar_width=0.2
    rat1 = -np.diff(hiVCP, axis=0)
    rat2 = -np.diff(loVCP, axis=0)
    plt.bar(np.array([-bar_width/2, +bar_width/2]), np.array([rat1, rat2]).squeeze(), bar_width, color=('c','m'), zorder=1)
    m=np.max(np.abs(plt.ylim()))
    plt.ylim([-m,m])
    adjustPlot(ax, fuzzyzero=True)
    plt.xticks(np.array([-bar_width/2, +bar_width/2]), ['High Variance\nTraining', 'Low Variance\nTraining'])
    plt.ylabel('State Evidence Ratio')
    plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')


#%%
if __name__ == '__main__':
    main()