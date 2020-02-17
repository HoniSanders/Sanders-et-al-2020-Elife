# -*- coding: utf-8 -*-
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from matplotlib import cm

def stripFilename(filepath):
    return os.path.basename(filepath).split('.')[0]


def binstarts2binedges(vec):
    vec = vec.reshape(vec.size,) # works with column or row vectors
    d = np.diff(vec)[0]     # assume all diffs are same
    return np.append(vec - d/2, vec[-1] + d/2)
    

def collapseIndices(c):
    # input is list of indices
    # output is equivalent list of indices such that there are no skipped indices (0,1,...,n) 
    #   and indices are in order of first appearance
    ret = -1*np.ones((c.shape), dtype=int)
    for i in range(ret.size):
        if ret[i] == -1:
            n = max(ret) + 1
            ret[c==c[i]] = n
    return ret

def adjustPlot(ax, fuzzyzero=False):
    #
    ax.grid(False)
    ax.set_facecolor((1,1,1))
    ax.spines['bottom'].set_color((0,0,0))
    ax.spines['left'].set_color((0,0,0))
    if fuzzyzero:
        xr = plt.xlim()
        yr = plt.ylim()
        nx, ny = 1000.,1000
        xgrid, ygrid = np.mgrid[xr[0]:xr[1]:(xr[1]-xr[0])/nx,yr[0]:yr[1]:(yr[1]-yr[0])/ny]
        im = stats.norm.pdf(ygrid, scale=5)
        plt.imshow(np.flipud(im.T), extent=(xr[0],xr[1],yr[0],yr[1]), cmap=cm.get_cmap('Greys'), 
                       vmin=0, vmax=2*np.max(im), zorder=0, aspect='auto')
        plt.xlim(xr)
        plt.ylim(yr)
    
params = {'alpha': 1e-1,
          'sigma': 0.1,
          'seed': 1,
          }