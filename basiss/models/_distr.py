import numpy as np
import pymc as pm
import aesara.tensor as at
from aesara.tensor import gammaln

#depricated
class Beta_sum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
    def logp(self, value):
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


def betaln(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logpow(x, m):
    """
    Calculates log(x**m) since m*log(x) will fail when m, x = 0.
    """
    # return m * log(x)
    return at.switch(at.eq(x, 0), at.switch(at.eq(m, 0), 0.0, -np.inf), m * at.log(x))


def beta_sum_logp(value, n, alpha, beta):
    beta_logp = logpow(value, alpha - 1) + logpow(1 - value, beta - 1) - betaln(alpha, beta)
    return beta_logp * n