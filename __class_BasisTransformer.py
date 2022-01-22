from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import torch as t

"""
=== Overview ===
By analogy to partial tomography, our quantum learning scheme requires multiple bases to 
learn the phases in the state expansion. The general idea will be to (a) randomly sample 
bases until a good one is found (i.e. one where the learning converges well) and then (b) 
perturb that basis slightly by a small rotation to generate a second good basis. It is 
possible to replace (b) with a second random sampling, but there is no need to do this 
unless we find that the above scheme doesn't learn the phases well enough because the two 
bases are too close to each other. Hopefully, that will not be the case.
"""


"""
Given a quantum state expressed in the z-basis, we apply a quantum circuit
operation to change the basis into a different one, using a parametrizaed
rotation via 1-gates. 
"""
class BasisTransformer:
    def __init__(self, states, updated_parameters):
        self.states = states
        self.L = states[0].num_qubits
        assert np.all([states[0].num_qubits == state.num_qubits for state in states])
        if type(updated_parameters).__module__ == t.__name__:
            self.p = updated_parameters.clone().cpu().detach().numpy()
        else:
            assert type(updated_parameters).__module__ == np.__name__
            self.p = updated_parameters
        self.__transform()
    
    def __make_qc(self):
        """
        Make the quantum circuit in qiskit to execute the transformation. Rotation
        performed with qiskit U-gates: https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html
        
        param_mat: L x L matrix of (theta, phi, lambda) parameters (each col)
        """
        qubits = QuantumRegister(self.L)
        qc = QuantumCircuit(qubits)
        
        for i in range(self.L):
            qc.u(*self.p[i], qubits[i])
        self.qc = qc
    
    def __transform(self):
        """Perform the change of basis for each state"""
        self.__make_qc()
        self.transformed_states = [state.copy().evolve(self.qc) for state in self.states]
    
    def transformed_states(self):
        return self.transformed_states
    
    def updated_dist(self):
        return [state.probabilities() for state in self.transformed_states]