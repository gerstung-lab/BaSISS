import numpy as np
import pymc3 as pm

class Beta_sum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
    def logp(self,value):
        return super().logp(value) * self.n

def beta_moments(data, axis=0):
    mean = np.mean(data, axis=axis)
    var = np.var(data, axis=axis)
    alpha = (mean*(1-mean)/var - 1)*mean
    beta = (mean*(1-mean)/var - 1)*(1-mean)
    return alpha, beta

def gamma_moments(data, axis=0):
    mean = np.mean(data, axis=axis)
    var = np.var(data, axis=axis)
    k = mean**2/var
    theta = var/mean
    return k, theta

def rho2sigma(rho):
    return np.log(1+np.exp(rho))