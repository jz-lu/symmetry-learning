"""
Find symmetries of the 7-qubit Rydberg chain
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
from Ryd_generator import ryd_ground_state
from math import pi

parser = argparse.ArgumentParser(description="Determine query complexity over number of qubits")
parser.add_argument("-d", "--depth", type=int, help='circuit block depth', default=3)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-n", "--nrun", type=int, help='number of runs to average over', default=10)
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("-x", "--xbasis", action='store_true', help='measure in x basis')
parser.add_argument("phase", type=str, choices=['Z2', 'Z3', 'DO', 'HE'])
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

CIRCUIT_DEPTH = args.depth
NUM_QUBITS = 7
NUM_BASES = args.bases
USE_REGULARIZER = False
LOSS_METRIC = KL
ESTIMATE = False
SAMPLE = False
NOISE_SCALE = 5
NRUN = args.nrun
OUTDIR = (args.out + '/') if args.out[-1] != '/' else args.out
OPS = None # MMD sigma parameter
PHASE = args.phase

losses = np.zeros(NRUN)
queries = np.zeros(NRUN)

print(f"Querying on L = {NUM_QUBITS}")
print(f"Phase = {PHASE}")
state = Statevector(ryd_ground_state(PHASE))

# Start the HQNet
bases = prepare_basis(state.num_qubits, num=NUM_BASES, init=pi/2 if args.xbasis else 0)
hqn = HQNet(state, bases, eta=1e-2, maxiter=1E15, disp=False,
            mode='Nelder-Mead', depth=CIRCUIT_DEPTH, 
            estimate=ESTIMATE, s_eps=NOISE_SCALE, 
            metric_func=LOSS_METRIC, ops=OPS, sample=SAMPLE, 
            regularize=USE_REGULARIZER)
dprint(f"[L={NUM_QUBITS}] Variational circuit:")
if args.verbose:
    print(hqn.view_circuit().draw())

# Find symmetries
param_shape = (state.num_qubits, CIRCUIT_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
param_dim = np.prod(param_shape)
proposed_syms = t.zeros((NRUN, param_dim))

for i in range(NRUN):
    print(f"Starting iteration {i}", flush=True)
    potential_sym, losses[i], queries[i] = hqn.find_potential_symmetry(print_log=args.verbose, include_nfev=True)
    proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)

print(f"[L={NUM_QUBITS}] Median loss: {np.median(losses)}, stdev: {np.std(losses)}")
print(f"[L={NUM_QUBITS}] Mean # queries: {np.mean(queries)}, stdev: {np.std(queries)}")
np.save(OUTDIR + f'syms_{PHASE}_L_{NUM_QUBITS}.npy', proposed_syms)

np.save(OUTDIR + f'losses_{NUM_QUBITS}_{PHASE}.npy', losses)
np.save(OUTDIR + f'queries_{NUM_QUBITS}_{PHASE}.npy', queries)
