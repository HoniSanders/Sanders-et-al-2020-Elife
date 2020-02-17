# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 9A of Sanders et al 2020
"""


from myutils import stripFilename, binstarts2binedges, adjustPlot
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
filename = stripFilename(__file__)
figInd = 0

try:
    filename = stripFilename(__file__)
except NameError:
    filename = 'script'
   
from partitionParams import partitions2activity, experiment

#%%

figInd=0
lw=0.7
alphas = [1e0, 1e-5, 1e-10]

figInd += 1
plt.figure(figInd)
plt.clf()

for alpha in alphas: 
    N=20
    rets = experiment(N=N, alpha=alpha)
    lps = rets[1]
    experiences = rets[7]
    posts = rets[8]

    plt.plot(np.arange(N)+1,-(lps[:,0]-lps[:,1]), label=r'$\alpha$ = {:.0e}'.format(alpha))
xlims=plt.xlim()
plt.ylim((-50/2,50/2))
adjustPlot(plt.gca(), fuzzyzero=True)
plt.legend()
plt.xlabel('# of experiences')
plt.ylabel('Negative Partition Evidence Ratio')

#%%
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
