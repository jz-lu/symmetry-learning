from __loss_funcs import KL
from __helpers import prepare_basis, qubit_expansion
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __class_HQNet import HQNet
from __class_PQC import PQC
import numpy as np
import torch as t
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator
from qiskit import transpile, QuantumRegister
from qiskit.providers.aer.noise import pauli_error, depolarizing_error
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer.noise import NoiseModel
import argparse

parser = argparse.ArgumentParser(description="Noisy GHZ HQNSL")
parser.add_argument("-p", '--prob', type=int, help='probability (units of 1e-4)', default=1)
parser.add_argument('--id', type=int, help='ID number', default=1)
args = parser.parse_args()

NUM_QUBITS = 3
qreg = QuantumRegister(NUM_QUBITS)
STATE_TYPE = 'GHZ'
PROBABILITY = 0.001 * args.prob

PROB_DEPOL = PROBABILITY
PROB_RESET = PROBABILITY
PROB_MEAS = 0
PROB_GATE1 = PROBABILITY

# Prepare noisy and noiseless GHZ State 
if STATE_TYPE == 'GHZ':
    from GHZ_generator import GHZ_state_circuit
    noiseless_state = Statevector.from_int(0, 2**NUM_QUBITS)
    qc = GHZ_state_circuit(L=NUM_QUBITS, qreg=qreg)
elif STATE_TYPE == 'mGHZ':
    from mGHZ_generator import mGHZ_state_circuit
    noiseless_state = Statevector.from_int(0, 2**NUM_QUBITS)
    qc = mGHZ_state_circuit(L=NUM_QUBITS)
elif STATE_TYPE == 'Cluster':
    from cluster_generator import cluster_state_circuit
    noiseless_state = Statevector.from_int(0, 2**(NUM_QUBITS**2))
    qc = cluster_state_circuit(NUM_QUBITS)
    NUM_QUBITS = NUM_QUBITS**2
else:
    raise TypeError(f"Invalid state type '{STATE_TYPE}' specified.")
noiseless_state = noiseless_state.evolve(qc)

bases = prepare_basis(noiseless_state.num_qubits)
DEPTH = 0
MAXITER = 1E4
num_bases = len(bases)
hqn = HQNet(noiseless_state, bases, eta=1e-2, maxiter=MAXITER, disp=False,
            mode='Nelder-Mead', depth=DEPTH, 
            noise=2, state_prep_circ=qc, qreg=qreg, 
            error_prob=PROBABILITY, 
            metric_func=KL, regularize=False)

# Find the symmetries of the noiseless and noisy states.
param_shape = (noiseless_state.num_qubits, DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
NRUN = 5
param_dim = np.prod(param_shape)
proposed_syms = t.zeros((NRUN, param_dim)) # first dim is for the 3 types of noise
losses = np.zeros(NRUN)

total_loss = 0
for j in range(NRUN):
    potential_sym, loss, regularizer_loss = hqn.find_potential_symmetry(print_log=True)
    proposed_syms[j] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
    potential_sym = potential_sym.reshape(param_shape)
    total_loss += loss
    losses[j] = loss
print(f"Average loss: {total_loss / NRUN}")
np.save(f"./losses_noisy_{args.id}.npy", losses)

losses = np.zeros(NRUN)
for j, sym in enumerate(proposed_syms):
    losses[j] = np.mean([PQC(noiseless_state, depth=DEPTH, basis_param=basis_here, metric_func=KL, say_hi=False)\
            .evaluate_true_metric(sym)for basis_here in bases])
np.save(f"./losses_cv_{args.id}.npy", losses)
np.save(f"./syms_{args.id}.npy", proposed_syms)
print(f"{np.mean(losses)} with deviation {np.std(losses)}")



