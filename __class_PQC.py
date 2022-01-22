from __loss_funcs import KL
from __class_BasisTransformer import BasisTransformer
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np
import torch as t
from ___constants import CNET_TEST_SIZE, CNET_TRAIN_SIZE, SAMPLING_DENSITY
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
"""

class PQC:
    def __init__(self, state, basis_param):
        self.metric = KL #* change this if the metric changes
        self.state = BasisTransformer([state], basis_param).transformed_states[0]
        self.L = state.num_qubits
        
    def __Q_th(self, p):
        """
        Apply the quantum circuit in qiskit corresponding to Q_theta with parameters p.
        For the moment, it is also just a bunch of single-qubit rotations.
        This will eventually change to a richer parametrization of circuits.
        """
        qubits = QuantumRegister(self.L)
        Q_th = QuantumCircuit(qubits)
        p_sqrt = int(np.sqrt(p.size()[0]))
        p = t.reshape(p, (p_sqrt, p_sqrt))
        if type(p).__module__ == t.__name__:
            p = p.clone().cpu().detach().numpy()
        
        for i in range(self.L):
            Q_th.u(*p[i], qubits[i])
        return self.state.copy().evolve(Q_th)
        
    def evaluate_true_metric(self, p):
        """
        Calculate the metric loss of Q_th(p)|state> against reference |state>,
        where p parametrizes the circuit Q_th.
        """
        return self.metric(self.state.probabilities(), self.__Q_th(p).probabilities())
    
    def states_from_rand_params(self, sz):
        """
        Generate `sz` random input parameters, which inputs into the CNet, 
        sampled from X ~ 2 * Pi * DUnif(n), whwere n = SAMPLING_DENSITY.
        """
        dataset = t.zeros(sz, 3*self.L + 1) # + 1 for the output value
        params = t.randint(0, SAMPLING_DENSITY, (sz, self.L*3)) * 2 * pi / SAMPLING_DENSITY
        dataset[:,:-1] = params # the last column of the dataset stores the output loss metric
        true_metric = t.tensor([self.evaluate_true_metric(param) for param in params]) # target value for CNet
        dataset[:,-1] = true_metric
        return dataset
  
    def generate_train_test(self, train_size=CNET_TRAIN_SIZE, test_size=CNET_TEST_SIZE):
        """
        Generate the train and test datasets for neural network training.
        A convenience function mostly for unit-testing the CNet.
        """
        train_data = self.states_from_rand_params(train_size)
        test_data = self.states_from_rand_params(test_size)
        return train_data, test_data