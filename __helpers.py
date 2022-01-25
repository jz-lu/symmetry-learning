# * Miscellaneous helper functions.
import numpy as np
import numpy.random as npr
from math import pi
import torch as t

def qubit_expansion(L):
    """Returns list of all L-qubits in z basis in lexicographical order"""
    assert isinstance(L, int) and L > 0
    d2b = np.vectorize(np.binary_repr)
    return d2b(np.arange(2**L), L)

def rand_basis(L):
    """Returns a random basis choice parametrized by (theta, phi, lambda) for L qubits"""
    return (2 * pi * t.rand(3)).repeat((L, 1))