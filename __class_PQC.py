from ___constants import CNET_TEST_SIZE, CNET_TRAIN_SIZE, SAMPLING_DENSITY
from __loss_funcs import KL
from __class_BasisTransformer import BasisTransformer
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np
import torch as t
from math import pi

"""
PQC: parametrized quantum circuit. Having chosen (in advance) a parametrization 
for a family of Q-circuits, we map the parameter to its corresponding circuit below. 

`basis_param` is a Lx3 matrix of parameters specifying a tensor product of rotations in the 
L-tensored Bloch sphere. The exact parametrization is specified by Qiskit's U-gate.

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
    def __init__(self, state, basis_param=None, metric_func=KL, say_hi=True):
        self.metric = metric_func
        self.state = state
        self.L = state.num_qubits
            
        if type(basis_param).__module__ == t.__name__:
            self.bp = basis_param.clone().cpu().detach().numpy()
        elif type(basis_param).__module__ == np.__name__:
            self.bp = basis_param
        self.bp = -self.bp # *= -1 since measurement ~= inverted rotation
        
        # Obtain distribution of state when measured in the given basis.
        if basis_param is not None:
            self.basis_dist = BasisTransformer([state], basis_param).updated_dist()[0]
        else:
            self.basis_dist = state.probabilities()
        
        if say_hi:
            print("Parametrized quantum circuit initialized.")
        
    def __Q_th(self, p):
        """
        Apply the quantum circuit in qiskit corresponding to Q_theta with parameters p.
        For the moment, it is also just a bunch of single-qubit rotations.
        This will eventually change to a richer parametrization of circuits.
        * Parametrization-dependent function!
        """
        qubits = QuantumRegister(self.L)
        Q_th = QuantumCircuit(qubits)
        p_sqrt = int(np.sqrt(p.shape[0]))
        p = t.reshape(p, (p_sqrt, p_sqrt))
        if type(p).__module__ == t.__name__:
            p = p.clone().cpu().detach().numpy()
        
        # Quantum parametrization -- DYNAMIC (change based on circuit param)
        for i in range(self.L):
            Q_th.u(*p[i], qubits[i])
        
        # Measurement in the basis -- STATIC (do not change regardless of parametrization)
        for i in range(self.L):
            Q_th.u(*self.bp[i], qubits[i])
        
        return self.state.copy().evolve(Q_th)
        
    def evaluate_true_metric(self, p):
        """
        Calculate the metric loss of Q_th(p)|state> against reference |state>,
        where p parametrizes the circuit Q_th.
        """
        return self.metric(self.basis_dist, self.__Q_th(p).probabilities())
    
    def true_metric_on_params(self, p_list):
        """Same as `evaluate_true_metric` but over a list of parameters"""
        assert len(p_list) > 0, "Parameter list empty"
        return [self.evaluate_true_metric(p) for p in p_list]
    
    def gen_rand_data(self, sz, include_metric=True, localize=False, requires_grad=False):
        """
        Generate `sz` random input parameters (just one axis rotation for now), 
        which inputs into the CNet, 
        sampled from X ~ 2 * Pi * DUnif(n), whwere n = SAMPLING_DENSITY.
        
        * Parametrization-dependent function!
        """
        n_param = 3 * self.L
        if include_metric:
            dataset = t.zeros(sz, n_param + 1)
            params = t.randint(0, SAMPLING_DENSITY, (sz, self.L*3)) * 2 * pi / SAMPLING_DENSITY
            if localize:
                params[:,7:] = 0
            dataset[:,:-1] = params # the last column of the dataset stores the output loss metric
            true_metric = t.tensor([self.evaluate_true_metric(param) for param in params]) # target value for CNet
            dataset[:,-1] = true_metric
        else:
            dataset = t.randint(0, SAMPLING_DENSITY, (sz, n_param)) * 2 * pi / SAMPLING_DENSITY
            if requires_grad:
                dataset.requires_grad_(True)
        return dataset

    def generate_train_test(self, train_size=CNET_TRAIN_SIZE, test_size=CNET_TEST_SIZE):
        """
        Generate the train and test datasets for neural network training.
        A convenience function mostly for unit-testing the CNet.
        """
        train_data = self.gen_rand_data(train_size, localize=False) # localize is a debugging parameter
        test_data = self.gen_rand_data(test_size)
        return train_data, test_data