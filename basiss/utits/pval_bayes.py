import numpy as np


def pval_bayes(X, Y, by=1):
    Y = Y + 1e-10
    n = X.shape[0]
    pr = 1 - np.stack([(X / Y > by).sum(axis=0) / n, (X / Y < 1 / by).sum(axis=0) / n], axis=1).max(axis=1)
    # pr[pr > 0.5] = 1 - pr[pr > 0.5]
    pr[pr == 0] = 1 / n
    return pr
