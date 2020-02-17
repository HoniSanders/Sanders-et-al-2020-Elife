#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: honi

Core inference machinery
Top level classes are at the bottom: 
    Animal uses WorldModel, which in turn uses distributions defined above
"""

import numpy as np
#from inference import Binom, Gauss, VonMises, Mixed, Similarity, GenericMixed
from scipy import stats
from scipy.special import gammaln, gamma, logsumexp
import pycircstat
from myutils import collapseIndices, binstarts2binedges
from matplotlib import pyplot as plt

# everything is in log space

#%%
# Distributions not included in scipy.stats
class multivariate_t():
    def __init__(self, df=None, loc=np.array([[0]]), scale=np.array([[1]])):
        self.mu_0 = loc
        self.Sigma = scale
        self.df = df
        # Sigma can be square matrix corresponding to length of mu_o
        assert(self.Sigma.shape[0] == self.Sigma.shape[1])
        assert(self.Sigma.shape[0] == self.mu_0.shape[0])

    def pdf(self, x, df=None, loc=None, scale=None):
        # x: NxK matrix.  N observations with K-dimensions each
        if loc is None:
            loc = self.mu_0
        if scale is None:
            Sigma = self.Sigma
        else:
            Sigma = scale
        if df is None:
            df = self.df
            assert(df is not None)
        
        x = np.array(x)
        if not x.size:
            return 0.0  # log likelihood of 0 if y is empty
        if len(x.shape)==1:
            x = x.reshape(1,-1)
        N = len(x)      # number of observations
        K = len(x[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(x[i]) == K

        SigmaInv = np.linalg.inv(Sigma)
        SigmaDet = np.linalg.det(Sigma)
        
        ret = np.zeros((N,1))*np.nan
        for i in range(N):
            numerator = gamma(1.0 * (K + df) / 2.0)  
            xSxT = np.matmul(np.matmul(x[i,:] - loc, SigmaInv), (x[i,:] - loc).T)
            denominator = (
                gamma(1.0 * df / 2.0) * 
                np.power(df * np.pi, 1.0 * K / 2.0) *  
                np.power(SigmaDet, 1.0 / 2.0) * 
                np.power(1. + (1./df) * xSxT, 1. * (K + df) / 2.)
                )
            ret[i] = (numerator/denominator).squeeze()
        if N == 1:
            return ret[0]
        else:
            return ret
    
    def logpdf(self, x, df=None, loc=None, scale=None):
        return np.log(self.pdf(x,df,loc,scale))
    
    
def generalizedgamma(p, a):
    templist = [gamma((2*a-i)/2) for i in range(p)]
    return np.pi**(p*(p-1)/4)*np.prod(np.array(templist))


def generalizedgammaln(p, a):
    templist = [gammaln((2*a-i)/2) for i in range(p)]
    return (p*(p-1)/4)*np.log(np.pi) + np.sum(np.array(templist))



#%%
# The following classes perform inference over observations from a single cluster
## Likelihoods
class Generative:
    def posteriorPredictive(self, yi, y):
        print('posteriorPredictive not implemented.')

    
    def nullPosteriorPredictive(self, yi, prior=None):
        # value of posteriorPredictive if no prior experiences given
        #return 0
        if prior is None:
            return self.marginalLikelihood(yi)
        else:
            return self.marginalLikelihood(yi, prior=prior)

    
    def marginalLikelihood(self, y):
        print('marginalLikelihood not implemented.')

    
    def fullLikelihood(self, yi, clusteredExperiences, orient=False):
        # orient is only implemented for VonMises
        ret=np.empty((len(clusteredExperiences) + 1,1))
        orientation = np.array(ret)
        for c in range(len(clusteredExperiences)):
            if orient:
                ret[c], orientation[c] = self.posteriorPredictive(yi, clusteredExperiences[c], orient=orient)
            else:                
                ret[c] = self.posteriorPredictive(yi, clusteredExperiences[c])
        ret[-1] = self.nullPosteriorPredictive(yi)
        if orient:
            orientation[-1] = 0
            return ret, orientation
        else:
            return ret    
    
    
    def partitionLikelihood(self, clusteredExperiences):
        # returns a value for the log likelihood of a given partition
        ll = 0.0        # log likelihood
        for c in range(len(clusteredExperiences)):
            y = clusteredExperiences[c]
            if not np.array(y).size:
                return 0.0  # log likelihood of 0 if y is empty
            N = len(y)      # number of observations
            K = len(y[0])   # number of dimensions of each observation
            for i in range(N):
                assert len(y[i]) == K
            ll += self.marginalLikelihood(y)
        return ll

    


priortype = 'Normal-Wishart'
#priortype = 'Normal-Gamma'
class Gauss(Generative):
    def __init__(self, mu_0=0, sigma_0=1):
        #mu_0 = 0        # prior mean
        #sigma_0 = 10    # prior std dev
        self.prior = stats.norm(loc=mu_0, scale=sigma_0)
        self.beta_0 = 0.01
        self.alpha_0 = 0.01
#        self.alpha_0 = .1     # Gamma prior for 1/sigma^2
#        self.beta_0 = self.alpha_0*sigma_0 #.1      # Gamma prior for 1/sigma^2
        self.kappa_0 = 0.001 #np.sqrt(1/2/sigma_0)     # Normal-Gamma prior for mu


    def logpdf(self, x, loc=0, scale=1):
        return stats.norm.logpdf(x, loc=loc, scale=scale)


    def posteriorPredictive(self, yi, y, return_dist=False, prior=priortype):
        # log likelihood of observation yi coming from the same cluster as all of the observations contained in list y
        # assuming gaussian likelihood with ? prior
        y = np.array(y)
        if not y.size:
            return self.nullPosteriorPredictive(yi, prior=prior)
        if not y.size or len(y.shape)==1 or len(y.shape)==0:
            y = y.reshape(1,-1)
        N = len(y)      # number of observations
        K = len(y[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(y[i]) == K
        yi=np.array(yi)
        if len(yi.shape) > 1 and yi.shape[0] == 1:  # yi i
            yi = yi[0]
        try:
            assert(len(yi) == K)
        except TypeError:
            # if yi is zero-dimensional
            yi = np.array([yi,])

        # #  what prior to use; normal estimates mu; normal-gamma estimates mu and sigma
        # prior = 'Normal-Wishart' #'Normal-Gamma' #'Normal'    
        # prior = 'Normal-Gamma' #'Normal'    
        # prior = 'Normal'    
        if prior == 'Normal':
            mu_0 = self.prior.mean()       # prior mean
            sigma_0 = self.prior.std()     # prior std dev
            ll = 0.0        # log likelihood
            if N == 1:      # if only one data point
                eps = sigma_0   # use prior std dev for sample std dev
            else:
                eps = 1e-10
            mus = np.mean(y, axis=0)                     # mean of past data for each dim
            sigmas = np.maximum(np.std(y, axis=0), eps)  # std dev of past data for each dim
            sigmas_n = np.sqrt(1/(N/sigmas**2 + 1/sigma_0**2))          # posterior std dev
            mus_n = sigmas_n**2 * (mu_0/sigma_0**2 + N*mus/sigmas**2)   # posterior mean
            ll = np.sum(self.logpdf(yi, loc=mus_n, scale=np.sqrt(sigmas_n**2 + sigmas**2)))
            if return_dist:
                return ll, stats.norm(loc=mus_n, scale=np.sqrt(sigmas_n**2 + sigmas**2))
            return ll
        elif prior == 'Normal-Gamma':
            mu_0 = self.prior.mean()       # prior mean
            sigma_0 = self.prior.std()     # prior std dev
            alpha_0 = self.alpha_0
            beta_0 = self.beta_0
            kappa_0 = self.kappa_0
    
            alpha_n = alpha_0 + N/2
            beta_n = beta_0 + 1/2*N*y.std(axis=0)**2 + \
                    (kappa_0*N*(y.mean(axis=0)-mu_0)**2)/(2*(kappa_0+N))
            kappa_n = kappa_0 + N
            mu_n = (kappa_0*mu_0+N*y.mean(axis=0))/(kappa_0+N)
            sigma = np.sqrt(beta_n*(kappa_n+1)/(alpha_n*kappa_n))
            ll = np.sum(stats.t.logpdf(yi,2*alpha_n, loc=mu_n, scale=sigma))
            if return_dist:
                return ll, stats.t(2*alpha_n, loc=mu_n, scale=sigma)
            return ll    
        elif prior == 'Normal-Wishart':
            # K is number of dimensions of obs
            mu_0 = self.prior.mean()            # prior mean
            T_0 = np.eye(K)*self.beta_0*2 #*self.prior.std()    # prior covariance
            kappa_0 = self.kappa_0
            nu_0 = self.alpha_0*2 #K                    # degrees of freedom for wishart
    
            y_mean = y.mean(axis=0)
            if y.shape[0]==1:
                S=np.zeros(T_0.shape)
            else:
                S = (N-1)*np.cov(y.T) #N*(np.std(y)**2)
            nu_n = nu_0 + N
            kappa_n = kappa_0 + N
            mu_n = (kappa_0*mu_0+N*y_mean)/(kappa_0+N)
            T_n = T_0 + S + kappa_0*N/(kappa_0+N)*np.matmul((mu_0-y_mean).T, mu_0-y_mean)
            
            dist = multivariate_t(df=nu_n-K+1, loc=mu_n, scale=T_n*(kappa_n+1)/(kappa_n*(nu_n-K+1)))
            ll = dist.logpdf(yi)
            if return_dist:
                return ll, dist
            return ll    

        
    def marginalLikelihood(self, y, prior=priortype):
        # log likelihood of all the observations coming from a single cluster
        # assuming gaussian likelihood with Normal-Gamma prior
        y = np.array(y)
        if not y.size:
            return 0.0  # log likelihood of 0 if y is empty
        if not y.size or len(y.shape)==1 or len(y.shape)==0:
            y = y.reshape(1,-1)
        N = len(y)      # number of observations
        K = len(y[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(y[i]) == K

        if prior == 'Normal':
            mu_0 = self.prior.mean()       # prior mean for mu
            sigma_0 = self.prior.std()     # prior std dev for mu
            if N == 1:      # if only one data point
                eps = sigma_0   # use prior std dev for sample std dev
            else:
                eps = 1e-10
            mus = y.mean(axis=0)
            sigmas = np.maximum(np.std(y, axis=0), eps)  # std dev of past data for each dim
            term1 = np.log(sigmas/((np.sqrt(2*np.pi)*sigmas)**N * np.sqrt(N*sigma_0**2+sigmas**2)))
            term2 = -np.sum(y**2,axis=0)/(2*sigmas**2) - mu_0**2/(2*sigma_0**2)
            term3 = (sigma_0**2*N**2*mus**2)/sigmas**2 + sigmas**2*mu_0**2/sigma_0**2 + 2*N*mus*mu_0
            term4 = 2*(N*sigma_0**2+sigmas**2)
            return np.sum(term1+term2+term3/term4)
        elif prior == 'Normal-Gamma':
            mu_0 = self.prior.mean()       # prior mean for mu
            sigma_0 = self.prior.std()     # prior std dev for mu
            alpha_0 = self.alpha_0
            beta_0 = self.beta_0
            kappa_0 = self.kappa_0
    
            alpha_n = alpha_0 + N/2
            beta_n = beta_0 + 1/2*N*y.std(axis=0)**2 + \
                    (kappa_0*N*(y.mean(axis=0)-mu_0)**2)/(2*(kappa_0+N))
            kappa_n = kappa_0 + N
            ret = np.log(gamma(alpha_n)/gamma(alpha_0)*beta_0**alpha_0/beta_n**alpha_n * \
                    np.sqrt(kappa_0/kappa_n)*(2*np.pi)**(-N/2))
            return np.sum(ret)
        elif prior == 'Normal-Wishart':
            # K is number of dimensions of obs
            mu_0 = self.prior.mean()            # prior mean
            T_0 = np.eye(K)*self.beta_0*2 #*self.prior.std()    # prior covariance
            kappa_0 = self.kappa_0
            nu_0 = self.alpha_0*2 #K                    # degrees of freedom for wishart
    
            y_mean = y.mean(axis=0)
            if y.shape[0]==1:
                S=np.zeros(T_0.shape)
            else:
                S = (N-1)*np.cov(y.T) #N*(np.std(y)**2)
            nu_n = nu_0 + N
            kappa_n = kappa_0 + N
            mu_n = (kappa_0*mu_0+N*y_mean)/(kappa_0+N)
            T_n = T_0 + S + kappa_0*N/(kappa_0+N)*np.matmul((mu_0-y_mean).T, mu_0-y_mean)
            
            a = (N*K/2.)*np.log(1./np.pi)
            b = generalizedgammaln(K, nu_n/2.)
            b2 = generalizedgammaln(K, nu_0/2.)
            c = np.log(np.linalg.det(T_0)**(nu_0/2.) / np.linalg.det(T_n)**(nu_n/2.))
            d = np.log((kappa_0/kappa_n)**(K/2.))
            return a + b - b2 + c + d




class VonMises(Generative):
    def __init__(self):
        beta_0 = .01
        alpha_0 = .01
#        beta_0 = .02
#        alpha_0 = .02        
        kappa_0 = .001 #np.sqrt(1/2/sigma_0)     # Normal-Gamma prior for mu
        mu_0 = 0       

        
        nMus = 200
        mus = np.arange(-np.pi,np.pi,np.pi/nMus*2)
        step=1
        lambdamax = 180
        lambdas = np.arange(step/2,lambdamax,step)
        mv, lv = np.meshgrid(mus,lambdas, indexing='ij')
        logprior = stats.norm.logpdf(mv, loc=mu_0, scale=np.sqrt(1/(kappa_0*lv))) \
                    + alpha_0*np.log(beta_0)+(alpha_0-1)*np.log(lv)-beta_0*lv-gammaln(alpha_0)
 #       logprior = alpha_0*np.log(beta_0)+(alpha_0-1)*np.log(lv)-beta_0*lv-gammaln(alpha_0)
        logprior -= logsumexp(logprior)                    
        
        self.beta_0 = beta_0
        self.alpha_0 = alpha_0
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.mus, self.lambdas = mus, lambdas
        self.mv, self.lv = mv, lv
        self.logprior = logprior
        
        self.marginalLikelihoodCache = dict()
        


    def logpdf(self, x, loc=0, scale=1):
        return stats.vonmises.logpdf(x, kappa=scale, loc=loc)
    
    
    def orient(self, yi, y):
        lls=dict()
        for offset in np.linspace(-np.pi, np.pi, np.pi/100):
            lls[offset] = self.posteriorPredictive(yi-offset, y)
        return max(lls, key=lls.get)

   
    def posteriorPredictive(self, yi, y, return_params=False, orient=False): 
        y = np.array(y)
        if not y.size:
            return self.nullPosteriorPredictive(yi)
        if not y.size or len(y.shape)==1 or len(y.shape)==0:
            y = y.reshape(1,-1)
        N = len(y)      # number of observations
        K = len(y[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(y[i]) == K
        yi=np.array(yi)
        if len(yi.shape) > 1 and yi.shape[0] == 1:  # yi i
            yi = yi[0]
        try:
            assert(len(yi) == K)
        except TypeError:
            # if yi is zero-dimensional
            yi = np.array([yi,])
        
        # p(yi|D) = int_params p(yi|params) p(params|D)
        # p(params|D) = p(D|params) p(params|params_0) / p(D)
        # p(yi|params) is predictive likelihood (need full matrix)
        # construct p(params|D) matrix from p(D|params) matrix (marginal likelihood likelihood matrix)
        #   and p(params|params_0) (marginal likelihood prior matrix)
        # p(D) is marginal likelihood
        _, pYk, pParams, pYkGivenParams, mv, lv = self.marginalLikelihood(y, return_dists=True)
        pParamsGivenYk = np.array(pYkGivenParams)
        for k in range(K):
            pParamsGivenYk[k] = pParams + pYkGivenParams[k] - pYk[k]
                
        mus, lambdas = self.mus, self.lambdas
        logdmudlambda = np.log(1./len(mus) * 1./len(lambdas))
        pYikGivenParams = np.empty((K, len(mus), len(lambdas)))
        # pdf(x, lambda, mu) = pdf(x-mu, lambda), so instead of looping over mu,
        # we calculate pdf(x-mus, lambda)
        for l in range(len(lambdas)):
            for k in range(K):
                # multiply by dmu dlambda when converting from pdf to discrete probability
                pYikGivenParams[k,:,l] = stats.vonmises.logpdf(yi[k]-mus, lambdas[l]) + logdmudlambda
   
        if orient:
            pYikGivenYkWithOffset = np.empty((K, len(mus)))
            for k in range(K):
                p = pParamsGivenYk[k]
                for offset in range(len(mus)):
                    l = np.roll(pYikGivenParams[k], offset, axis=0)
                    pYikGivenYkWithOffset[k,offset] = logsumexp(p + l)
            # sum over k
            pYiGivenYWithOffset = np.sum(pYikGivenYkWithOffset, axis=0)
            bestOffset = np.argmax(pYiGivenYWithOffset)
            dmu = mus[1]-mus[0]
            orientation = np.mod(-bestOffset*dmu, 2*np.pi)
            return pYiGivenYWithOffset[bestOffset], orientation
        else:
            # p(yi|D) = prod_k p(yi[k]|D[k]), equivalently sum of logs
            pYikGivenYk = np.array(yi)
            for k in range(K):
                pYikGivenYk[k] = logsumexp(pParamsGivenYk[k] + pYikGivenParams[k])            
            return np.sum(pYikGivenYk)
        
    
    def marginalLikelihood(self, y, return_dists=False):
        y = np.array(y)
        if not y.size:
            return 0.0  # log likelihood of 0 if y is empty
        if not y.size or len(y.shape)==1 or len(y.shape)==0:
            y = y.reshape(1,-1)
        N = len(y)      # number of observations
        K = len(y[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(y[i]) == K
        
        mus, lambdas = self.mus, self.lambdas
        logdmudlambda = np.log(1./len(mus) * 1./len(lambdas))
        pParams = self.logprior
        try:
            pYkGivenParams = self.marginalLikelihoodCache[y.tobytes()]    
        except KeyError:  
            pYkGivenParams = np.empty((K, len(mus), len(lambdas)))
            # pdf(x, lambda, mu) = pdf(x-mu, lambda), so instead of looping over mu,
            # we calculate pdf(x-mus, lambda)
            for l in range(len(lambdas)):
                for k in range(K):
                    yikxmu = np.tile(y[:,k,np.newaxis], len(mus))
                    # multiply by dmu dlambda when converting from pdf to discrete probability
                    tmp = stats.vonmises.logpdf(yikxmu-mus, lambdas[l]) + logdmudlambda
                    # sum over yi in y
                    pYkGivenParams[k, :, l] = np.sum(tmp, axis=0)                    
            self.marginalLikelihoodCache[y.tobytes()] = pYkGivenParams
        pYk = np.empty((K,))
        for k in range(K):
            pYk[k] = logsumexp(pParams + pYkGivenParams[k])
        # sum over k
        pY = np.sum(pYk)
        if return_dists:
            plot_dists = False
            if plot_dists:
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(pParams, aspect='auto', vmin=-50, vmax=0, origin='lower',
                           extent=[np.min(lambdas), np.max(lambdas), np.min(mus), np.max(mus)])
                plt.ylabel('mu')
                plt.title('prior over parameters')
                plt.colorbar()
                plt.subplot(2,2,2)
                plt.imshow(pYkGivenParams[0], aspect='auto', vmin=-50, vmax=0, origin='lower',
                           extent=[np.min(lambdas), np.max(lambdas), np.min(mus), np.max(mus)])
                plt.colorbar()
                plt.xlabel('lambda')
                plt.title('likelihood of data given parameters')
                plt.subplot(2,2,3)
                plt.imshow(pParams + pYkGivenParams[0], aspect='auto', vmin=-50, vmax=0, origin='lower',
                           extent=[np.min(lambdas), np.max(lambdas), np.min(mus), np.max(mus)])
                plt.ylabel('mu')
                plt.xlabel('lambda')
                plt.title('posterior of data')
                plt.colorbar()
        if return_dists:            
            return pY, pYk, pParams, pYkGivenParams, self.mv, self.lv
        return pY

        


class GenericMixed(Generative):
    def __init__(self, distList):
        # distList is list of distribution instances, e.g. (Gauss(), Binom(), Gauss(mu_0=10))
        self.distList = distList
        
    def posteriorPredictive(self, yi, y):
        if not np.array(y).size:
            return self.nullPosteriorPredictive(yi)
        if not y.size or len(y.shape)==1 or len(y.shape)==0:
            y = y.reshape(1,-1)
        N = len(y)      # number of observations
        K = len(y[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(y[i]) == K
        if len(yi.shape) > 1 and yi.shape[0] == 1:  # yi i
            yi = yi[0]
        try:
            assert(len(yi) == K)
        except TypeError:
            # if yi is zero-dimensional
            yi = np.array([yi,])
            
        ll = 0
        for k in range(K):  # for each dimension of the observation
            yik = yi[k]
            yk = y[:,k].reshape(-1,1)
            ll += self.distList[k].posteriorPredictive(yik, yk)
        return ll
   

    def marginalLikelihood(self, y):
        if not np.array(y).size:
            return 0.0  # log likelihood of 0 if y is empty
        if not y.size or len(y.shape)==1 or len(y.shape)==0:
            y = y.reshape(1,-1)
        N = len(y)      # number of observations
        K = len(y[0])   # number of dimensions of each observation
        for i in range(N):
            assert len(y[i]) == K
            
        ll = 0
        for k in range(K):  # for each dimension of the observation
            yk = y[:,k].reshape(-1,1)
            ll += self.distList[k].marginalLikelihood(yk)
        return ll

    
    
    



#%%
## Priors over partitions
class CRP:
    # Chinese Restaurant Process
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        
    def prior(self, c):
        # c is a list of cluster indices (length = # of observations)
        # returns array with prior probability (log space) of adding a new 
        # observation to each cluster index (length = # of clusters + 1)
        # normalized so sum(exp(return array)) = 1
        c = np.array(c, dtype=int)
        unique, counts = np.unique(c, return_counts=True)
        if np.sum(counts) == 0:
            return np.array([0.])
        d = dict(zip(unique, counts))
        C = int(np.max(c))   # highest index cluster; used as # of clusters
        counts = np.zeros((C+1,1))
        for key in d:
            counts[key] = d[key]
        counts = np.vstack((counts, self.alpha))
        return np.log(counts / float(np.sum(counts)))
    
    def partition_prob(self, c):
        # c is a list of cluster indices (length = # of observations)
        # returns a value which is the log probability of observing that partition
        c = np.array(c, dtype=int)
        unique, counts = np.unique(c, return_counts=True)
        if np.sum(counts) == 0:
            return 0
        C = int(np.max(c))   # highest index cluster; used as # of clusters
        N = np.sum(counts)
        countswalpha = np.concatenate((counts, np.array([self.alpha])))
        logp = C*np.log(self.alpha) + np.sum(gammaln(countswalpha)) - gammaln(N+self.alpha)
        return logp
    
    def rvs(self, size=1):
        ret = -1*np.ones((size,))
        ret[0] = 0
        for i in range(1,size):
            unique, counts = np.unique(ret, return_counts=True)
            unique, counts = unique[unique!=-1], counts[unique!=-1]
            p = counts/(np.sum(counts)+self.alpha)
            a = np.where(np.random.random()<=np.cumsum(p))
            if type(a) is tuple:
                a=a[0]
            if a.size==0:
                ret[i] = np.max(unique)+1
            else:
                ret[i] = unique[np.max(a)]
        return ret
   
    

def bellNumber(n):
    # # of partitions of set size n
    bell = [[0 for i in range(n+1)] for j in range(n+1)]
    bell[0][0] = 1
    for i in range(1, n+1):
        # Explicitly fill for j = 0
        bell[i][0] = bell[i-1][i-1]
        # Fill for remaining values of j
        for j in range(1, i+1):
            bell[i][j] = bell[i-1][j-1] + bell[i][j-1] 
    return bell[n][0]



class Flat:
    # Alternative prior that doesn't favor clusters with more past observations
    def __init__(self, scale=1):
        # scale = 1 means that the prior is flat,
        # scale = 0 means that prior is nonexistent
        self.scale = scale
        
        
    def prior(self, c):
        # c is a list of cluster indices
        c = np.array(c, dtype=int)
        C = int(np.max(c))   # highest index cluster; used as # of clusters
        counts = np.zeros((C+1,1))      # log(1) = 0 equal probability for any cluster
        counts = np.vstack((counts, 0)) # log(1) = 0 probability for novel cluster
        if self.scale:
            counts -= logsumexp(counts)     # normalize by sum
            return counts
        else:
            return counts       # prior is just log(1) = 0 for everything
       
    def partition_prob(self, c):
        # TODO: check if this works properly
        # c is a list of cluster indices
        c = np.array(c, dtype=int)
        C = int(np.max(c))   # highest index cluster; used as # of clusters
        if self.scale:
            return np.log(1./bellNumber(C))     # = log(1/# of possible partitions)
        else:
            return 0.                           # = log(1)
 




#%%
# The following classes perform inference over partitions of observations 
# using the generative distributions implemented above for each individual cluster
## Internal Model (used by Animal below, uses distributions above)
class WorldModel:
    # examples:
    # CRP-Gaussian
    # Flat-Similarity
    def __init__(self, clusterModel, observationModel):
        # clusterModel and observationModel should be instances, not classes
        # clusterModel must have prior() method
        # observationModel must have fullLikelihood() method
        self.clusterModel = clusterModel
        self.prior = self.clusterModel.prior
        self.observationModel = observationModel
        self.likelihood = self.observationModel.fullLikelihood
        
    def belief(self, yi, y, c, normalize=True, orient=False):
        # yi is a 1xK array of current observation
        # y is NxK array of past observations
        # c is Nx1 array of past cluster assignments
        p = self.prior(c)
        ce = experiences2ClusteredExperiences(y,c)
        if orient:
            l, orientations = self.likelihood(yi, ce, orient=True)
        else:
            l = self.likelihood(yi, ce)
        m = l[l > -np.Inf].min()
        l[np.isinf(l)] = (m-1)*10
        # adding prior and posterior in log space is equivalent to multiplying in scalar space
        post = p + l
        #print(np.hstack((p,l,post)))
        if normalize:
            post -= logsumexp(post)     # normalize by sum
        if orient:
            return post, orientations
        else:
            return post
    
    def partition_prob(self, y, c):
        # y is NxK array of past observations
        # c is Nx1 array of past cluster assignments
        # returns a value which is the log probability of that partition given the data
        p = self.clusterModel.partition_prob(c)
        ce = experiences2ClusteredExperiences(y,c)
        l = self.observationModel.partitionLikelihood(ce)
        #print(p,l)
        post = p + l
        return post
    
    
    
    
## Utilities
def experiences2ClusteredExperiences(experiences, clusterAssignments):
    # y is NxK array of past observations
    # c is Nx1 array of past cluster assignments
    clusteredExperiences = []
    # len-0 means a single 1d obs, len-1 means a single shape[0]-d obs, len-2 means shape[0] shape[1]-d obs
    experiences = np.array(experiences)
    if len(experiences.shape) == 0 or len(experiences.shape) == 1: # only 1 experience
        return [experiences.reshape(1,-1)]
    K = experiences.shape[1]   # # of dimensions per observation
    N = experiences.shape[0]     # # of experiences
    for i in range(N):
        while clusterAssignments[i] >= len(clusteredExperiences):
            clusteredExperiences.append(np.array([]))
        if K == 1:  # 1-d observations
            if clusteredExperiences[int(clusterAssignments[i])].size==0:    # if this is first experience in cluster
                clusteredExperiences[int(clusterAssignments[i])] = np.array([experiences[i]])
            else:
                clusteredExperiences[int(clusterAssignments[i])] = \
                    np.vstack((clusteredExperiences[int(clusterAssignments[i])], experiences[i]))
        else:   
            if clusteredExperiences[int(clusterAssignments[i])].size==0:    # if this is first experience in cluster
                clusteredExperiences[int(clusterAssignments[i])] = np.array([experiences[i,:]]) 
            else:
                clusteredExperiences[int(clusterAssignments[i])] = \
                    np.vstack((clusteredExperiences[int(clusterAssignments[i])], experiences[i,:]))
    return clusteredExperiences

    

## Top level class
## Agent
class Animal:
    experiences = np.array([])           # T-length list of K-dimensional experiences, i.e. T rows, K columns
    clusterAssignments = np.array([],dtype=int)   # T-length list of cluster assignment indices, i.e. T rows, 1 column
    K = np.nan                              # # of dimensions of experiences
    
    def __init__(self, alpha=0.1, clusterModelClass=CRP, observationModelClass=Gauss, 
                 priorExperiences=None, priorClusterAssignments=None, worldModel=None):
        # clusterModelClass must have prior() method
        # observationModelClass must have fullLikelihood() method
        if worldModel is None:
            self.alpha = alpha                  # tendency to open new clusters
            cM = clusterModelClass(self.alpha)
            oM = observationModelClass()
            # prior outputs vector of log probabilities of length nClusters + 1
            # likelihood should have two args: current observation and clusteredExperiences
            self.worldModel = WorldModel(cM, oM)
        else:
            self.worldModel = worldModel
        if priorExperiences is not None:
            assert priorExperiences.shape[0] == len(priorClusterAssignments)
            self.experiences = priorExperiences
            priorClusterAssignments = np.reshape(priorClusterAssignments, (len(priorClusterAssignments),-1))
            self.clusterAssignments = priorClusterAssignments.astype(int)
    # attributes:
    # self.alpha
    # self.worldModel
    # self.experiences
    # self.clusterAssignments
    # self.K

    
    def __copy__(self):
        return type(self)(worldModel=self.worldModel, 
                          priorExperiences=self.experiences, 
                          priorClusterAssignments=self.clusterAssignments)
    
    def experience(self, yi, calculate_belief=True, orient=False, normalize=True):
        # saves experience
        # returns belief state of cluster assignment probabilities for this experience
        try:    # turn scalar into array
            len(yi)
        except TypeError:
            yi = np.array([yi])
        if len(yi.shape) == 1:     # turn 1d array into 2d array
            yi = np.array([yi])
        # add experience to self.experiences
        shape = self.experiences.shape
        if shape[-1] == 0:         # if there are no past observations
            self.experiences = yi
            self.clusterAssignments = np.array([0], dtype=int)      # assign the first experience to the first cluster
            return np.array([0])    # 100% probability that this experience comes from the first cluster
        else:
            self.K = shape[-1]
            if yi.size > 1:
                yi = np.squeeze(yi)
            assert len(yi) == self.K

            if calculate_belief:
                if orient:
                    b, orientations = self.belief(yi, orient=orient, normalize=normalize)      # belief state of cluster assignment probabilities for current experience
                else:
                    b = self.belief(yi, normalize=normalize)      # belief state of cluster assignment probabilities for current experience
                c = int(np.argmax(b))   # MAP assignment of cluster index, given previous cluster assignments (local MAP)
                if orient:
                    yi = yi - orientations[c]
            else:
                c = 0

            self.experiences = np.vstack((self.experiences, yi))
            self.clusterAssignments = np.vstack((self.clusterAssignments, c))
            if calculate_belief:
                if orient:
                    return b, orientations
                else:
                    return b
        
    def belief(self, yi, clusterAssignments=None, orient=False, normalize=True):
        # log space, sum of exp(belief) should = 1
        if clusterAssignments is None:
            clusterAssignments = self.clusterAssignments
        if orient:
            b, orientations = self.worldModel.belief(yi, self.experiences, clusterAssignments, orient=orient, normalize=normalize)
            return b, orientations
        else:
            return self.worldModel.belief(yi, self.experiences, clusterAssignments, normalize=normalize)
    
    def partition_prob(self, c):
        # c is Nx1 array of past cluster assignments
        return self.worldModel.partition_prob(self.experiences, c)

    def gibbsUpdate(self, i):
        c = np.delete(self.clusterAssignments, i, 0)
        y = np.delete(self.experiences, i, 0)
        yi = self.experiences[i]
        post = self.worldModel.belief(yi, y, c)
        cu = np.cumsum(np.exp(post))
        np.testing.assert_almost_equal(cu[-1], 1)
        ind = np.searchsorted(cu > np.random.random(), 1)   # sample from the posterior
        self.clusterAssignments[i] = int(ind)
        self.clusterAssignments = collapseIndices(self.clusterAssignments)


    def reassessAssignments(self, return_cInferredList=False, make_plots=False):
        cInferredLast = np.nan*self.clusterAssignments
        nSteps = 20
        j = 0
        burnin = 10
        nSamples = 30
        while not np.array_equal(self.clusterAssignments, cInferredLast) or j >= nSteps*burnin:
            # if we've tried 10 times and haven't converged, gather nSamples samples from posterior
            cInferredLast = np.array(self.clusterAssignments)
            for ind in range(nSteps):       # do 10 MCMC steps
                updateOrder = np.random.permutation(self.experiences.shape[0])
                for i in updateOrder:
                    self.gibbsUpdate(i)
                j += 1
            print((np.squeeze(self.clusterAssignments), j))
            if j == nSteps*burnin:
                cInferredList = np.empty((nSamples, np.max(self.clusterAssignments.shape)))
            if j >= nSteps*burnin:
                cInferredList[int(j/nSteps)-burnin,:] = np.squeeze(self.clusterAssignments)
                if j == nSteps*(nSamples+burnin-1):
                    nExp = cInferredList.shape[1]
                    sims = np.empty((nExp,nExp))
                    
                    for col1 in range(nExp):
                        for col2 in range(col1):
                            sims[col1,col2] = np.mean(cInferredList[:,col1]==cInferredList[:,col2])
                            sims[col2,col1] = sims[col1,col2]
                        sims[col1,col1] = 1
                    if make_plots:
                        plt.figure()
                        vrange = (0, 1)
                        cmap = plt.get_cmap('jet')
                        plt.imshow(sims, vmin=vrange[0], vmax=vrange[1], cmap=cmap, aspect='auto')
                        plt.show()
                    if return_cInferredList:
                        return cInferredList
                    else:
                        break
        # if broken because of early convergence
        if return_cInferredList:
            return np.tile(self.clusterAssignments, (nSamples,1))

    