"""
Generate global regularization score on 3-GHZ state.
"""
from __helpers import prepare_basis, param_to_unitary
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __loss_funcs import KL
from __class_HQNet import HQNet
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import argparse
from qiskit.quantum_info import Statevector
from GHZ_generator import GHZ_state_circuit
from PCA import classify_sym

parser = argparse.ArgumentParser(description="Compute SG on n-GHZ state")
parser.add_argument("-L", "--L", type=int, help='number of qubits', default=3)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("epochs", type=int, help='number of superepochs')
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

CIRCUIT_DEPTH = 0
NUM_QUBITS = args.L
NUM_BASES = args.bases
USE_REGULARIZER = True
LOSS_METRIC = KL
ESTIMATE = False
SAMPLE = False
NOISE_SCALE = 5
NEPOCH = args.epochs
OUTDIR = (args.out + '/') if args.out[-1] != '/' else args.out
OPS = None # MMD sigma parameter
MIN_N0, MAX_N0, NUM_N0 = 0, 5e3, 25
N0s = np.linspace(MIN_N0, MAX_N0, NUM_N0+1)[1:]
MAXITER = 1e4
scores = np.zeros(NUM_N0)

# Prepare state noiselessly
state = Statevector.from_int(0, 2**NUM_QUBITS)
qc = GHZ_state_circuit(L=NUM_QUBITS)
state = state.evolve(qc)
dprint("State preparation circuit:")
dprint(qc)

bases = prepare_basis(state.num_qubits, num=NUM_BASES)

# Find symmetries
param_shape = (state.num_qubits, CIRCUIT_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
param_dim = np.prod(param_shape)
proposed_syms = t.zeros((NUM_N0, NEPOCH, 2, param_dim))

losses = np.zeros((NUM_N0, NEPOCH, 2))
reglosses = np.zeros((NUM_N0, NEPOCH, 2, NUM_BASES))
unitaries_prods = np.zeros((NUM_N0, NEPOCH, 2, NUM_QUBITS, 2, 2)) # |N0| x epochs x nruns x NUM_QUBITS x matrix dim = 2x2

for pj, N0 in enumerate(N0s):
    for epoch in range(NEPOCH):
        # Reset the HQNet
        hqn = HQNet(state, bases, eta=1e-2, maxiter=1E4, disp=False,
                mode='Nelder-Mead', depth=CIRCUIT_DEPTH, 
                estimate=ESTIMATE, s_eps=NOISE_SCALE, 
                metric_func=LOSS_METRIC, ops=OPS, sample=SAMPLE, 
                jump=USE_REGULARIZER, checkpoint=int(N0))
        for i in range(2):
            potential_sym, losses[pj,epoch,i], reglosses[pj,epoch,i] = hqn.find_potential_symmetry(print_log=args.verbose)
            proposed_syms[pj,epoch,i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
            
            # Pushforward from parameter space to SU(2)^((x) 3)
            unitaries_prods[pj,epoch,i] = param_to_unitary(proposed_syms[pj,epoch,i].reshape((NUM_QUBITS, -1)).numpy())
            
    print(f"[N0={N0}] Median loss: {np.median(losses[pj])}, stdev: {np.std(losses[pj])}")
    
unitaries_prods = np.around(unitaries_prods, 4)
np.save(OUTDIR + 'reglosses.npy', reglosses)
np.save(OUTDIR + 'losses.npy', losses)
np.save(OUTDIR + 'unitaries.npy', unitaries_prods)

sym_labels = np.zeros((NUM_N0, NEPOCH, 2))
for pj in range(NUM_N0):
    for epoch, unitary_prod_set in enumerate(unitaries_prods[pj]):
        sym_labels[pj,epoch] = np.array([classify_sym(np.abs(unitary_prod)) for unitary_prod in unitary_prod_set])
np.save(OUTDIR + 'labels.npy', sym_labels)

# Compute the score function
for pj in range(NUM_N0):
    score = 0
    for m1, m2 in sym_labels:
        if (m1 in [1,3] and m2 in [2,4]) or (m2 in [1,3] and m1 in [2,4]):
            score += 1
    scores[pj] = score / NEPOCH
np.save(OUTDIR + 'SG.npy', scores)

# Plot the score
plt.clf()
plt.title("Global regularization score on 3-GHZ")
plt.xlabel(r"$N_0$")
plt.ylabel(r"$S_G(N_0)$")
plt.plot(N0s, scores, c='k')
plt.savefig(OUTDIR + 'SG.pdf')
plt.show()
