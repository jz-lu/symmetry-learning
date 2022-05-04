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
from math import pi

parser = argparse.ArgumentParser(description="Determine query complexity over circuit depth")
parser.add_argument("-L", "--L", type=int, help='number of qubits', default=5)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-n", "--nrun", type=int, help='number of runs to average over', default=10)
parser.add_argument("-x", "--xbasis", action='store_true', help='measure in x basis')
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("state", type=str, help='family of states to learn on', choices=['GHZ', 'XY'])
parser.add_argument("depth", type=int, help='circuit block depth')
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

DEPTH = args.depth
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
    
bases = prepare_basis(state.num_qubits, num=NUM_BASES, init=pi/2 if args.xbasis else 0)

losses = np.zeros(NRUN)
queries = np.zeros(NRUN)

print(f"Querying on d = {DEPTH}")

# Start the HQNet
hqn = HQNet(state, bases, eta=1e-2, maxiter=1E15, disp=False,
            mode='Nelder-Mead', depth=DEPTH, 
            estimate=ESTIMATE, s_eps=NOISE_SCALE, 
            metric_func=LOSS_METRIC, ops=OPS, sample=SAMPLE, 
            regularize=USE_REGULARIZER)
dprint(f"[d={DEPTH}] Variational circuit:")
if args.verbose:
    print(hqn.view_circuit().draw())

# Find symmetries
param_shape = (state.num_qubits, DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
param_dim = np.prod(param_shape)
proposed_syms = t.zeros((NRUN, param_dim))

for i in range(NRUN):
    potential_sym, losses[i], queries[i] = hqn.find_potential_symmetry(print_log=args.verbose, include_nfev=True)
    proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
    potential_sym = potential_sym.reshape(param_shape)
print(f"[d={DEPTH}] Median loss: {np.median(losses[DEPTH])}, stdev: {np.std(losses[DEPTH])}")
print(f"[d={DEPTH}] Mean # queries: {np.mean(queries[DEPTH])}, stdev: {np.std(queries[DEPTH])}")
np.save(OUTDIR + f'syms_{STATE_TYPE}_depth_{DEPTH}.npy', proposed_syms)

np.save(OUTDIR + f'losses_{DEPTH}_{STATE_TYPE}.npy', losses)
np.save(OUTDIR + f'queries_{DEPTH}_{STATE_TYPE}.npy', queries)

# # Plot the scaling complexity
# avgs = np.mean(queries, axis=1)
# stdevs = np.std(queries, axis=1)
# x = np.arange(DEPTH+1)
# COLOR = 'darkblue'

# plt.clf()
# plt.title(fr"$d$-query complexity of {STATE_TYPE}")
# plt.xlabel("Depth")
# plt.plot(x, avgs, c=COLOR)
# plt.fill_between(x, avgs - stdevs, avgs + stdevs, color=COLOR, alpha=0.2)
# plt.savefig(OUTDIR + f"nfev_{STATE_TYPE}.pdf")
