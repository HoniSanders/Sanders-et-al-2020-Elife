# -*- coding: utf-8 -*-
"""
@author: Honi Sanders

# generates Fig. 9C of Sanders et al 2020
# compares a probe along a dimension that had either high or low variance during training
# fig1 is scatter plot of experiences
# fig2 is bar graph of the posterior for each partition for various alpha values (Fig. 9C)
# fig3 is a line graph similar to fig2 that varies alpha continuously 
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


sd = params['sigma']        # std dev of features
distance = 2    # distance between different features
K = 2           # # of features

np.random.seed(params['seed'])
sds = sd*np.ones((1,K))
mus = np.zeros((1,K))

morphSquare = stats.norm(loc=np.array([-distance/2, -distance/2]), scale=sds)
morphCircle = stats.norm(loc=np.array([+distance/2, -distance/2]), scale=sds)
whiteCircle = stats.norm(loc=np.array([+distance/2, +distance/2]), scale=sds)

#%%
# ms wc ms wc ms wc ms wc ms mc ms mc ms mc 

day1 = np.vstack((morphSquare.rvs(),
                  whiteCircle.rvs(),
                  morphSquare.rvs(),
                  whiteCircle.rvs(),
                  morphSquare.rvs(),
                  whiteCircle.rvs(),
                  ))
day4 = np.vstack((morphSquare.rvs(),
                  whiteCircle.rvs(),
                  morphSquare.rvs(),
                  morphCircle.rvs(),
                  morphSquare.rvs(),
                  morphCircle.rvs(),
                  ))
day5 = np.vstack((morphSquare.rvs(),
                  morphCircle.rvs(),
                  morphSquare.rvs(),
                  morphCircle.rvs(),
                  morphSquare.rvs(),
                  morphCircle.rvs(),
                  ))
observations = np.vstack((day1,
                          day1,
                          day1,
                          day4,
                          day5,
                          day5,
                          ))

#%%
figInd += 1
#%%
fig = plt.figure(figInd)
fig.clear()
plt.plot(observations[:,0],observations[:,1],'.')
plt.title('Observations')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
#%%
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
#%%

# rows are experiences, columns are different partitions
day1partitions = np.array([[0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [1, 1, 0],
                           ])
day4partitions = np.array([[0, 0, 0],
                           [1, 1, 0],
                           [0, 0, 0],
                           [2, 1, 0],
                           [0, 0, 0],
                           [2, 1, 0],
                           ])
day5partitions = np.array([[0, 0, 0],
                           [2, 1, 0],
                           [0, 0, 0],
                           [2, 1, 0],
                           [0, 0, 0],
                           [2, 1, 0],
                           ])
partitions = np.vstack((day1partitions,
                        day1partitions,
                        day1partitions,
                        day4partitions,
                        day5partitions,
                        day5partitions
                        ))
    


#%%
alphas = np.power(10.,np.arange(1,-50,-1))
posts = np.empty((len(alphas),3))*np.nan
for idx, alpha in enumerate(alphas):
    animal = Animal(alpha, observationModelClass=Gauss)
    for i in range(observations.shape[0]):
        animal.experience(observations[i,:])
        
    for i in range(3):      # for each partition
        posts[idx, i] = animal.partition_prob(partitions[:,i])

posts
#%%
figInd += 1
#%%
fig = plt.figure(figInd)
fig.clear()
bar_width = 0.2
idxs = np.array([0, 15, 30])+1#+5
idxs = np.array([0, 20, 40])+1#+5
for i in range(3):
    plt.bar(np.array([1,2,3])+(i-1)*bar_width, posts[idxs,i], bar_width)
    #plt.plot(np.array([1,2,3])+(i-1)*bar_width/10, posts[idxs,i], 'x')
labels = []
for idx in idxs:
    labels.append(r'$\alpha$ = ' + str(alphas[idx]))
adjustPlot(plt.gca(), fuzzyzero=False)
plt.xticks(np.array([1,2,3]), labels)
plt.ylabel('Unnormalized Log Posterior')
plt.legend(('All Separate', '"Correct"', 'All Together'), loc='lower center')
#%%
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')


#%%
figInd += 1
#%%
fig = plt.figure(figInd)
fig.clear()
plt.plot(np.log10(alphas),posts)
plt.xlabel('Log10(alpha)')
plt.ylabel('Unnormalized Log Posterior')
plt.legend(('All Separate', '"Correct"', 'All Together'))
#%%
plt.savefig('figures/' + filename + '-' + str(figInd) + '.pdf', bbox_inches='tight')
