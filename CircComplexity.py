"""
Display mean and variance of circuit complexity of the 4-qubit cluster
state, 4-qubit GHZ state, and 4-qubit XY Hamiltonian GS.
"""
from Scaling import STATE_TYPE
from __helpers import prepare_basis, param_to_unitary
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __loss_funcs import KL
from __class_HQNet import HQNet
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import argparse
from qiskit.quantum_info import Statevector

parser = argparse.ArgumentParser(description="Show symmetries of 3-GHZ in 2D DR space")
parser.add_argument("-d", "--depth", type=int, help='maximum circuit block depth', default=5)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-r", "--reg", action='store_true', help='use regularizer')
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("nrun", type=int, help='number of symmetries to find')
parser.add_argument("state", type=str, choices=['GHZ', 'XY', 'Cluster'])
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

MAX_DEPTH = args.depth
NUM_QUBITS = 4
NUM_BASES = args.bases
USE_REGULARIZER = args.reg
LOSS_METRIC = KL
ESTIMATE = False
SAMPLE = False
NOISE_SCALE = 5
NRUN = args.nrun
OUTDIR = (args.out + '/') if args.out[-1] != '/' else args.out
STATE_TYPE = args.state
assert NRUN >= 50, f"Number of runs should be at least 50 to be robust, but is {NRUN}"

def dprint(msg):
    if args.verbose:
        print(msg)

# Prepare the desired state
if STATE_TYPE == 'GHZ':
    from GHZ_generator import GHZ_state_circuit
    state = Statevector.from_int(0, 2**NUM_QUBITS)
    qc = GHZ_state_circuit(L=NUM_QUBITS)
    dprint(qc)
    state = state.evolve(qc)
elif STATE_TYPE == 'XY':
    from XY_generator import xy_ground_state
    state = Statevector(xy_ground_state(NUM_QUBITS).numpy())
elif STATE_TYPE == 'Cluster':
    from cluster_generator import cluster_state_circuit
    state = Statevector.from_int(0, 2**(2**2))
    qc = cluster_state_circuit(2)
    dprint(qc)
    state = state.evolve(qc)
    
bases = prepare_basis(state.num_qubits, num=NUM_BASES)
losses = np.zeros((MAX_DEPTH+1, NRUN))
queries = np.zeros((MAX_DEPTH+1, NRUN))

# Search for symmetries in progressive block depths
for CIRCUIT_DEPTH in range(1+MAX_DEPTH):
    hqn = HQNet(state, bases, eta=1e-2, maxiter=1E4*(CIRCUIT_DEPTH+1), disp=False,
                mode='Nelder-Mead', depth=CIRCUIT_DEPTH, 
                estimate=ESTIMATE, s_eps=NOISE_SCALE, 
                metric_func=LOSS_METRIC, ops=None, sample=SAMPLE, 
                jump=USE_REGULARIZER)
    dprint(f"[d={CIRCUIT_DEPTH}] Variational circuit:")
    if args.verbose:
        hqn.view_circuit().draw()
    
    param_shape = (state.num_qubits, CIRCUIT_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
    param_dim = np.prod(param_shape)
    # proposed_syms = t.zeros((NRUN, param_dim))
    
    for i in range(NRUN):
        _, losses[CIRCUIT_DEPTH, i], queries[CIRCUIT_DEPTH, i] = hqn.find_potential_symmetry(print_log=args.verbose, include_nfev=True)
       #  proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
    
    print(f"[d={CIRCUIT_DEPTH}] Median loss: {np.median(losses[CIRCUIT_DEPTH])}, stdev: {np.std(losses[CIRCUIT_DEPTH])}")
    print(f"[d={CIRCUIT_DEPTH}] Median loss: {np.median(queries[CIRCUIT_DEPTH])}, stdev: {np.std(queries[CIRCUIT_DEPTH])}")

np.save(OUTDIR + "losses.npy", losses)
np.save(OUTDIR + "queries.npy", queries)

# Plot the data as a bar graph
bottom_95 = round(NRUN * 0.95) # filter bad runs
losses = np.sort(losses, axis=0)[:,:bottom_95]

x = np.arange(MAX_DEPTH+1)
y = np.mean(losses, axis=1)
err = np.stdev(losses, axis=1)
plt.clf()
plt.title("Post-selected mean losses")
plt.xlabel("Block-depth")
plt.ylabel("QKL")
plt.bar(x, y, color='darkblue')
plt.errorbar(x, y, yerr=err, fmt='o', color='gray')
plt.savefig(OUTDIR + "circcomp.pdf")
