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

def prepare_basis(L, num=2, init=0):
    init_basis = t.zeros(L, 3) # z-basis
    init_basis[:,0] += init * t.ones(L)
    bases = [init_basis]
    basis_here = init_basis
    for _ in range(num-1):
        perturbation = t.zeros(L, 3)
        perturbation[:,0] += pi/10 * t.ones(L) # small rotation in one axis for each qubit
        basis_here = basis_here + perturbation
        bases.append(basis_here)
    return bases

def rand_basis(L):
    """Returns a random basis choice parametrized by (theta, phi, lambda) for L qubits"""
    return (2 * pi * t.rand(3)).repeat((L, 1))

def param_to_unitary(param):
    assert len(param.shape) == 2, "Function incompatible with circuit depth > 0"
    L = param.shape[0]
    unitaries = np.zeros((L, 2, 2), dtype=np.complex_)
    for i in range(L):
        theta, phi, lamb = param[i]
        unitaries[i,0,0] = np.cos(theta/2)
        unitaries[i,0,1] = -np.exp(1j*lamb) * np.sin(theta/2)
        unitaries[i,1,0] = np.exp(1j*phi) * np.sin(theta/2)
        unitaries[i,1,1] = np.exp(1j*(lamb+phi)) * np.cos(theta/2)
    return unitaries