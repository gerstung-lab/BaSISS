import numpy as np

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