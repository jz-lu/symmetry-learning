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

def qubit_retraction(bitstr):
    """Inverse operation to expansion: given bitstring, return decimal"""
    return int(bitstr, 2)

def prepare_basis(L, num=2):
    init_basis = t.zeros(L, 3) # z-basis
    bases = [init_basis]
    basis_here = init_basis
    for _ in range(num-1):
        perturbation = basis_here.clone()
        perturbation[:,npr.randint(0, 3)] += pi/10 * t.ones(L) # small rotation in one axis for each qubit
        basis_here = basis_here + perturbation
        bases.append(basis_here)
    return bases

def rand_basis(L):
    """Returns a random basis choice parametrized by (theta, phi, lambda) for L qubits"""
    return (2 * pi * t.rand(3)).repeat((L, 1))