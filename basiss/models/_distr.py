import numpy as np
import pymc as pm
import aesara.tensor as at
from aesara.tensor import gammaln


# deprecated
class Beta_sum(pm.Beta):
    def __init__(self, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def logp(self, value):
        return super().logp(value) * self.n


def beta_moments(data, axis=0):
    """Approximation of beta distributed variables

    Parameters
    ----------
    data : np.array
        Samples of beta distributed variables
    axis : int
        Dimension along which samples are stored

    Returns
    -------
    (np.array, np.array)
        Alpha, Beta parameters of variables
    """
    mean = np.mean(data, axis=axis)
    var = np.var(data, axis=axis)
    alpha = (mean * (1 - mean) / var - 1) * mean
    beta = (mean * (1 - mean) / var - 1) * (1 - mean)
    return alpha, beta


def gamma_moments(data, axis=0):
    """Approximation of gamma distributed variables

        Parameters
        ----------
        data : np.array
            Samples of gamma distributed variables
        axis : int
            Dimension along which samples are stored

        Returns
        -------
        (np.array, np.array)
            k, theta parameters of variables
        """
    mean = np.mean(data, axis=axis)
    var = np.var(data, axis=axis)
    k = mean ** 2 / var
    theta = var / mean
    return k, theta


def rho2sigma(rho):
    """Convert rho to sigma of a normal distribution

    Parameters
    ----------
    rho : np.array
        rho of the normal distribution
    Returns
    -------
    np.array
        sigma of the normal distribution
    """
    return np.log(1 + np.exp(rho))


def betaln(x, y):
    """Natural logarithm of the abs(beta function)

    Parameters
    ----------
    x : aesara.tensor
    y : aesara.tensor

    Returns
    -------
    aesara.tensor
        log(abs(beta func)))
    """
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def logpow(x, m):
    """Calculates log(x**m) since m*log(x) will fail when m, x = 0.

    Parameters
    ----------
    x aesara.tensor
    m aesara.tensor

    Returns
    -------
    aesara.tensor
        m * log(x)
    """
    return at.switch(at.eq(x, 0), at.switch(at.eq(m, 0), 0.0, -np.inf), m * at.log(x))


def beta_sum_logp(value, n, alpha, beta):
    """Likelihood of the product of beta distributions
    Parameters
    ----------
    value : aesara.tensor
        value at which logp is evaluated
    n : np.array
        Number of beta distributions
    alpha : aesara.tensor
        Alpha param of a beta distribution
    beta : aesara.tensor
        Beta param of a beta distribution

    Returns
    -------
    aesara.tensor
        n * beta.logp(value)
    """
    beta_logp = logpow(value, alpha - 1) + logpow(1 - value, beta - 1) - betaln(alpha, beta)
    return beta_logp * n
