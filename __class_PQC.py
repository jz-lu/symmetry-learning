from ___constants import (
    CNET_TEST_SIZE, CNET_TRAIN_SIZE, SAMPLING_DENSITY, 
    PARAM_PER_QUBIT_PER_DEPTH
)
from __loss_funcs import KL
from __class_BasisTransformer import BasisTransformer
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np
import torch as t
from math import pi

"""
PQC: parametrized quantum circuit. Having chosen (in advance) a parametrization 
for a family of Q-circuits, we map the parameter to its corresponding circuit below. 

`basis_param` is a L x depth+1 x PARAM_PER_QUBIT_PER_DEPTH matrix of parameters 
specifying a tensor product of rotations in the L-tensored Bloch sphere. 
The exact parametrization is specified by Qiskit's Ry and Rz gates.

Application of the circuit on a set of quantum states generates a dataset of states
evolved by the PQC. 

This class is static in the HQNet scheme, in the sense that we perform optimization on the 
parameter space, not the circuit architecture itself. That is, the map param -> circuit
is fixed; we simply want to find the parameter `param*` that produces a circuit leaving the
desired pristine state invariant, thereby finding a symmetry of the pristine state.

`metric_func`: choice of loss metric. Many others are available in `__loss_funcs.py`. 
Defaults to KL divergence.
"""

class PQC:
    def __init__(self, state, basis_param=None, metric_func=KL, depth=0, say_hi=True):
        assert depth >= 0 and isinstance(depth, int), f"Invalid circuit depth {depth}"
        self.metric = metric_func
        self.state = state
        self.L = state.num_qubits
        self.depth = depth
        self.n_param = self.L * PARAM_PER_QUBIT_PER_DEPTH * (self.depth + 1)
            
        if type(basis_param).__module__ == t.__name__:
            self.bp = basis_param.clone().cpu().detach().numpy()
        elif type(basis_param).__module__ == np.__name__:
            self.bp = basis_param
        else:
            print(type(basis_param).__module__)
        self.bp = -self.bp # *= -1 since measurement ~= inverted rotation
        
        # Obtain distribution of state when measured in the given basis.
        if basis_param is not None:
            self.basis_dist = BasisTransformer([state], basis_param).updated_dist()[0]
        else:
            self.basis_dist = state.probabilities()
        
        if say_hi:
            print("Parametrized quantum circuit initialized.")
    
    def __make_Q_th(self, p):
        """
        Make a quantum circuit in qiskit corresponding to parameters p. Currently, 
        the model used is a linear entanglement circuit with depth `self.depth`. This
        uses CNOT gates to entangle nearest neighbors `self.depth` times. Each sequence
        of entangling CNOT gates is sandwiched by universal single-qubit rotations.
        
        The most universal circuit of a given depth is a full entanglement, over every pair.
        We will restrict ourselves to a local version, for the time being at least.
        """
        qubits = QuantumRegister(self.L)
        Q_th = QuantumCircuit(qubits)
        assert p.shape[0] == self.n_param
        p = t.reshape(p, (self.L, self.depth+1, PARAM_PER_QUBIT_PER_DEPTH))
        if type(p).__module__ == t.__name__:
            p = p.clone().cpu().detach().numpy()
        
        # Quantum parametrization -- DYNAMIC (change based on circuit param)
        for i in range(self.L):
            Q_th.ry(p[i,0,0], qubits[i])
            Q_th.rz(p[i,0,1], qubits[i])
        for d in range(1, self.depth + 1):
            for i in range(self.L-1):
                Q_th.cx(i, i+1)
            for i in range(self.L):
                Q_th.ry(p[i,d,0], qubits[i])
                Q_th.rz(p[i,d,1], qubits[i])
        
        # Measurement in the basis -- STATIC (do not change regardless of parameterization)
        for i in range(self.L):
            Q_th.u(*self.bp[i], qubits[i])
            
        return Q_th
        
    def get_Q_th(self, p):
        """
        Get quantum circuit associated with parameter `p`.
        """
        return self.__make_Q_th(p)
    
    def __Q_th(self, p):
        """
        Apply the quantum circuit in qiskit corresponding to Q_theta with parameters p.
        """
        Q_th = self.__make_Q_th(p)
        return self.state.copy().evolve(Q_th)
    
    def get_circ(self, p):
        return self.__make_Q_th(p)
        
    def evaluate_true_metric(self, p):
        """
        Calculate the metric loss of Q_th(p)|state> against reference |state>,
        where p parametrizes the circuit Q_th.
        
        TODO | Currently, we are using the distribution from Qiskit. Ultimately, 
        TODO | we'd like to do it on a poly(L) sampling scheme. A priori it seems
        TODO | like a poly(L) scheme is "obviously" impossible, in which case a
        TODO | exp(L) scheme is also acceptable due to the fundamental dimensionality
        TODO | problem of the Hilbert space.
        """
        return self.metric(self.basis_dist, self.__Q_th(p).probabilities())
    
    def true_metric_on_params(self, p_list):
        """Same as `evaluate_true_metric` but over a list of parameters"""
        assert len(p_list) > 0, "Parameter list empty"
        return [self.evaluate_true_metric(p) for p in p_list]
    
    def gen_rand_data(self, sz, include_metric=True):
        """
        Generate `sz` random input parameters (just one axis rotation for now), 
        which inputs into the CNet, 
        sampled from X ~ 2 * Pi * DUnif(n), whwere n = SAMPLING_DENSITY.
        """
        if include_metric:
            dataset = t.zeros(sz, self.n_param + 1)
            params = t.randint(0, SAMPLING_DENSITY, (sz, self.n_param)) * 2 * pi / SAMPLING_DENSITY
            dataset[:,:-1] = params # the last column of the dataset stores the output loss metric
            true_metric = t.tensor([self.evaluate_true_metric(param) for param in params]) # target value for CNet
            dataset[:,-1] = true_metric
        else:
            dataset = t.randint(0, SAMPLING_DENSITY, (sz, self.n_param)) * 2 * pi / SAMPLING_DENSITY
        return dataset

    def generate_train_test(self, train_size=CNET_TRAIN_SIZE, test_size=CNET_TEST_SIZE):
        """
        Generate the train and test datasets for neural network training.
        A convenience function mostly for unit-testing the CNet.
        """
        train_data = self.gen_rand_data(train_size)
        test_data = self.gen_rand_data(test_size)
        return train_data, test_data