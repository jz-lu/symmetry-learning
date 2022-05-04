from qiskit import QuantumCircuit, QuantumRegister

def GHZ_state_circuit(L=3, qreg=None):
    # Create GHZ state circuit
    GHZ_circ = QuantumCircuit(QuantumRegister(L) if qreg is None else qreg)
    GHZ_circ.h(0)
    for i in range(1, L):
        GHZ_circ.cx(0, i)
    return GHZ_circ


