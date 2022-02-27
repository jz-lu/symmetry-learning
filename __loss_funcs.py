# * Loss functions for gradient descent / search
import numpy as np
import torch as t
import numpy.random as npr

# Not a loss function (just a helper)
def Gaussian(x, m, s):
    """Element-wise Gaussian map with mean m and deviation s"""
    return np.exp(-np.square(x-m) / (2 * s**2))

def KS(P1, P2):
    """Deprecated: KS Test on two discrete distributions"""
    assert len(P1) == len(P2)
    return np.abs(np.cumsum(P1) - np.cumsum(P2)).max()

def KL(P, Q, eps=1e-7):
    """KL divergence on two discrete distributions"""
    P = P + eps * np.ones_like(P)
    Q = Q + eps * np.ones_like(Q)
    return np.sum(P * np.log(P/Q))

def SymKL(P, Q, eps=1e-7):
    """Symmetrized KL divergence on two discrete distributions"""
    return 1/2 * (KL(P, Q, eps) + KL(Q, P, eps))

def Em_MMD(x, y, s=1):
    """Empirical (sample) MMD loss function with width parameter s"""
    assert len(x) == len(y), f"Samples x and y differ in size: {len(x)} vs {len(y)}"
    n = len(x)
    return sum([Gaussian(x[i], x[j], s) + Gaussian(y[i], y[j], s) - \
           2*Gaussian(x[i], y[j], s) for i in range(n) for j in range(n)]) / n**2

def MMD(P, Q, n=101, s=1):
    """
    Define the MMD loss over distributions as a random variable obtained by sampling
    from the distributions and computing the empirical MMD over the samples.
    """
    return Em_MMD(npr.choice(len(P), size=n, p=P), \
                  npr.choice(len(Q), size=n, p=Q), s=s)

