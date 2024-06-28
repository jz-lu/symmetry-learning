from qiskit import QuantumCircuit, QuantumRegister
from math import pi

def mGHZ_state_circuit(L=3):
    mGHZ_circ = QuantumCircuit(QuantumRegister(L))
    for i in range(0, L-1, 2):
        mGHZ_circ.h(i)
        mGHZ_circ.cx(i, i+1)
    for i in range(1, L-1, 2):
        mGHZ_circ.rx(pi/3, i)
        mGHZ_circ.ry(-pi/3, i)
    return mGHZ_circ