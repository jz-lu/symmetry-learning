import numpy as np
from __class_PQC import PQC
import qiskit.quantum_info as qi

def p2op(params):
    L, depth, _ = params.shape
    pqc = PQC(qi.Statevector.from_int(0, 2**L), depth=depth)
    return qi.Operator(pqc.get_circ(params))