# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 8 of Sanders et al 2020
# explores distribution of responses over the place field population
# fig1 gives examples of beta distributions with two different sets of parameters
# fig2 compares extent of partial remapping w extent of rate remapping
# fig3 compares extent of uncertainty w population heterogeneity
"""

import numpy as np
from scipy import stats
from scipy.misc import logsumexp

from myutils import stripFilename, adjustPlot
from matplotlib import pyplot as plt
plt.style.use('ggplot')
try:
    filename = stripFilename(__file__)
except NameError:
    filename = 'script'
figInd = 0

#%%

step = 0.01
x = np.arange(0, 1+step, step)

#%%
thresh = 0.15

x1 = np.arange(0, thresh+step, step)
x2 = np.arange(thresh, 1-thresh+step, step)
x3 = np.arange(1-thresh, 1+step, step)
def param_results(a,b):
    y2 = stats.beta.pdf(x2,a,b)
    
    sum1 = np.sum(stats.beta.pdf(x1,a,b))
    sum2 = np.sum(stats.beta.pdf(x2,a,b))
    sum3 = np.sum(stats.beta.pdf(x3,a,b))
    sum4 = sum1+sum2+sum3
    return [sum1/sum4, sum2/sum4, sum3/sum4], y2
    
#%%
figInd += 1
f=plt.figure(figInd)
f.clear()
ax = plt.gca()
a, b = 1, 7
plt.plot(x, stats.beta.pdf(x,a,b),linewidth=3)
print(param_results(a,b)[0])
a, b = 1.5, 1
plt.plot(x, stats.beta.pdf(x,a,b),linewidth=3)
print(param_results(a,b)[0])

ylims=ax.get_ylim()
plt.plot([0+thresh, 0+thresh], ylims,'--k')
plt.plot([1-thresh, 1-thresh], ylims,'--k')
ax.set_ylim([0, ylims[1]])
ax.set_xlim([0,1])
plt.xlabel('Rate Modulation')
plt.ylabel('Probability Density')
adjustPlot(ax)
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
  

#%%
maxER = 10
color1='slategrey'
color2='slategrey'

#%%
figInd += 1
f=plt.figure(figInd)
f.clear()
b=1
for a in range(1,maxER+1):
    p, rrs = param_results(a,b)
    plt.errorbar(p[2], np.average(x2,weights=stats.beta.pdf(x2,a,b)), color=color1, fmt='o', yerr=np.sqrt(stats.beta.stats(a,b,moments='v')))
a=1
for b in range(1,maxER+1):
    p, rrs = param_results(a,b)
    plt.errorbar(p[2], np.average(x2,weights=stats.beta.pdf(x2,a,b)), color=color2, fmt='o', yerr=np.sqrt(stats.beta.stats(a,b,moments='v')))
plt.xlabel('Extent of Partial Remapping')
plt.ylabel('Average Extent of Rate Remapping')
ax = plt.gca()
ax.set_xlim([0,1])
ax.set_ylim([0,1])
adjustPlot(ax)
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

#%% uncertainty vs heterogeneity
figInd += 1
f=plt.figure(figInd)
f.clear()
b=1
for a in range(1,maxER+1):
    p, rrs = param_results(a,b)
    plt.plot(1/np.exp(np.abs(a-b)), np.sqrt(stats.beta.stats(a,b,moments='v')), 'o', color=color1)
a=1
for b in range(1,maxER+1):
    p, rrs = param_results(a,b)
    plt.plot(1/np.exp(np.abs(a-b)), np.sqrt(stats.beta.stats(a,b,moments='v')), 'o', color=color2)
plt.xlabel('Uncertainty')
plt.ylabel('Heterogeneity')
ax = plt.gca()
ax.set_ylim([0,ax.get_ylim()[1]])
ax.set_xlim([0,ax.get_xlim()[1]])
adjustPlot(ax)
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')

