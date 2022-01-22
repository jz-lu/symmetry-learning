from __loss_funcs import KL
from __class_BasisTransformer import BasisTransformer
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np
import torch as t
from ___constants import CNET_TEST_SIZE, CNET_TRAIN_SIZE
from math import pi

"""
For two bases, generate train and test datasets along with the true metric value.
Specify a basis parametrization in addition to the state itself (assumed ot be in z-basis).
Apply a theta-parametrized quantum circuit Q_theta (Q_th) to the state.
"""
NUM_SAMPLES = 200
class CNetDataGenerator:
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
        
    def __true_loss(self, p):
        """
        Calculate the metric loss of Q_th(p)|state> against reference |state>,
        where p parametrizes the circuit Q_th.
        """
        return self.metric(self.state.probabilities(), self.__Q_th(p).probabilities())
    
    def __generate_some_data(self, sz):
        """
        Generate a random input parameter, which input to the CNet, 
        sampled from X ~ 2 * Pi * DUnif(n), whwere n = SAMPLING_DENSITY.
        """
        dataset = t.zeros(sz, 3*self.L + 1) # + 1 for the output value
        
        params = t.randint(0, NUM_SAMPLES, (sz, self.L*3)) * 2 * pi / NUM_SAMPLES
        dataset[:,:-1] = params
        
        true_metric = t.tensor([self.__true_loss(param) for param in params]) # target value for CNet
        dataset[:,-1] = true_metric
        return dataset
  
    def generate_train_test(self, train_size=CNET_TRAIN_SIZE, test_size=CNET_TEST_SIZE):
        """Generate the train and test datasets for neural network training"""
        train_data = self.__generate_some_data(train_size)
        test_data = self.__generate_some_data(test_size)
        return train_data, test_data