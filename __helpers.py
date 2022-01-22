# * Miscellaneous helper functions.
import numpy as np

def qubit_expansion(L):
    """Returns list of all L-qubits in z basis in lexicographical order"""
    assert isinstance(L, int) and L > 0
    d2b = np.vectorize(np.binary_repr)
    return d2b(np.arange(2**L), L)

