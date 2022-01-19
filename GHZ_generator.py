# Adapted from Kaiying's code to generate GHZ
from qiskit import QuantumCircuit, QuantumRegister

def GHZ_state_circuit():
    # Create GHZ state circuit
    GHZ_circ = QuantumCircuit(QuantumRegister(3))
    GHZ_circ.h(0); GHZ_circ.cx(0, 1); GHZ_circ.cx(0, 2)
    return GHZ_circ