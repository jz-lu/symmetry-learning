"""
Analyze the loss-functional query complexity over number of
qubits for the L-GHZ state, L-XY states.
"""
from __helpers import prepare_basis
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __loss_funcs import KL
from __class_HQNet import HQNet
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import argparse
from qiskit.quantum_info import Statevector
from GHZ_generator import GHZ_state_circuit

parser = argparse.ArgumentParser(description="Determine query complexity over circuit depth")
parser.add_argument("-d", "--depth", type=int, help='maximum circuit block depth', default=5)
parser.add_argument("-L", "--L", type=int, help='number of qubits', default=5)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-n", "--nrun", type=int, help='number of runs to average over', default=10)
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("state", type=str, help='family of states to learn on', choices=['GHZ', 'XY'])
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

MAX_CIRCUIT_DEPTH = args.depth
NUM_QUBITS = args.L
NUM_BASES = args.bases
USE_REGULARIZER = False
LOSS_METRIC = KL
ESTIMATE = False
SAMPLE = False
NOISE_SCALE = 5
NRUN = args.nrun
OUTDIR = (args.out + '/') if args.out[-1] != '/' else args.out
OPS = None # MMD sigma parameter
STATE_TYPE = args.state

# Prepare state noiselessly
if STATE_TYPE == 'GHZ':
    from GHZ_generator import GHZ_state_circuit
    state = Statevector.from_int(0, 2**NUM_QUBITS)
    qc = GHZ_state_circuit(L=NUM_QUBITS)
    dprint(qc)
    state = state.evolve(qc)
elif STATE_TYPE == 'XY':
    from XY_generator import xy_ground_state
    state = Statevector(xy_ground_state(NUM_QUBITS).numpy())
    
bases = prepare_basis(state.num_qubits, num=NUM_BASES)

losses = np.zeros((MAX_CIRCUIT_DEPTH+1, NRUN))
queries = np.zeros((MAX_CIRCUIT_DEPTH+1, NRUN))

for CIRCUIT_DEPTH in range(MAX_CIRCUIT_DEPTH+1):
    print(f"Querying on d = {CIRCUIT_DEPTH}")

    # Start the HQNet
    hqn = HQNet(state, bases, eta=1e-2, maxiter=1E5, disp=False,
                mode='Nelder-Mead', depth=CIRCUIT_DEPTH, 
                estimate=ESTIMATE, s_eps=NOISE_SCALE, 
                metric_func=LOSS_METRIC, ops=OPS, sample=SAMPLE, 
                regularize=USE_REGULARIZER)
    dprint(f"[d={CIRCUIT_DEPTH}] Variational circuit:")
    if args.verbose:
        print(hqn.view_circuit().draw())

    # Find symmetries
    param_shape = (state.num_qubits, CIRCUIT_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
    param_dim = np.prod(param_shape)
    proposed_syms = t.zeros((NRUN, param_dim))

    for i in range(NRUN):
        potential_sym, losses[CIRCUIT_DEPTH, i], queries[CIRCUIT_DEPTH, i] = hqn.find_potential_symmetry(print_log=args.verbose, include_nfev=True)
        proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
        potential_sym = potential_sym.reshape(param_shape)
    print(f"[d={CIRCUIT_DEPTH}] Median loss: {np.median(losses[CIRCUIT_DEPTH])}, stdev: {np.std(losses[CIRCUIT_DEPTH])}")
    print(f"[d={CIRCUIT_DEPTH}] Mean # queries: {np.mean(queries[CIRCUIT_DEPTH])}, stdev: {np.std(queries[CIRCUIT_DEPTH])}")
    np.save(OUTDIR + f'syms_{STATE_TYPE}_depth_{CIRCUIT_DEPTH}.npy', proposed_syms)
    
np.save(OUTDIR + f'losses_{STATE_TYPE}.npy', losses)
np.save(OUTDIR + f'queries_{STATE_TYPE}.npy', queries)

# Plot the scaling complexity
avgs = np.mean(queries, axis=1)
stdevs = np.std(queries, axis=1)
x = np.arange(MAX_CIRCUIT_DEPTH+1)
COLOR = 'darkblue'

plt.clf()
plt.title(fr"$d$-query complexity of {STATE_TYPE}")
plt.xlabel("Depth")
plt.plot(x, avgs, c=COLOR)
plt.fill_between(x, avgs - stdevs, avgs + stdevs, color=COLOR, alpha=0.2)
plt.savefig(OUTDIR + f"nfev_{STATE_TYPE}.pdf")
