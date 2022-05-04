"""
Display mean and variance of circuit complexity of the 4-qubit cluster
state, 4-qubit GHZ state, and 4-qubit XY Hamiltonian GS.
"""
from __helpers import prepare_basis, param_to_unitary
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __loss_funcs import KL
from __class_HQNet import HQNet
import matplotlib.pyplot as plt
from math import pi
import numpy as np
import torch as t
import argparse
from qiskit.quantum_info import Statevector

parser = argparse.ArgumentParser(description="Compute circuit complexity of a given state")
parser.add_argument("-d", "--depth", type=int, help='circuit block depth', default=5)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-r", "--reg", action='store_true', help='use regularizer')
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("-x", "--xbasis", action='store_true', help='measure in x basis')
parser.add_argument("nrun", type=int, help='number of symmetries to find')
parser.add_argument("state", type=str, help='oracle state', choices=['GHZ', 'XY', 'Cluster'])
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

DEPTH = args.depth
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
    
bases = prepare_basis(state.num_qubits, num=NUM_BASES, init=pi/2 if args.xbasis else 0)
losses = np.zeros(NRUN)
queries = np.zeros(NRUN)

# Search for symmetries 
hqn = HQNet(state, bases, eta=1e-2, maxiter=1E10*(DEPTH+1), disp=False,
            mode='Nelder-Mead', depth=DEPTH, 
            estimate=ESTIMATE, s_eps=NOISE_SCALE, 
            metric_func=LOSS_METRIC, ops=None, sample=SAMPLE, 
            jump=USE_REGULARIZER)
dprint(f"[d={DEPTH}] Variational circuit:")
if args.verbose:
    print(hqn.view_circuit().draw())

param_shape = (state.num_qubits, DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
param_dim = np.prod(param_shape)
# proposed_syms = t.zeros((NRUN, param_dim))

for i in range(NRUN):
    _, losses[i], queries[i] = hqn.find_potential_symmetry(print_log=args.verbose, include_nfev=True)
    #  proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)

print(f"[d={DEPTH}] Median loss: {np.median(losses[DEPTH])}, stdev: {np.std(losses[DEPTH])}", flush=True)
print(f"[d={DEPTH}] Median loss: {np.median(queries[DEPTH])}, stdev: {np.std(queries[DEPTH])}", flush=True)

np.save(OUTDIR + f"losses_{DEPTH}_{STATE_TYPE}.npy", losses)
np.save(OUTDIR + f"queries_{DEPTH}_{STATE_TYPE}.npy", queries)

# Plot the data as a bar graph
bottom_95 = round(NRUN * 0.95) # filter bad runs
losses = np.sort(losses, axis=0)[:,:bottom_95]
print("DONE!")

# x = np.arange(DEPTH+1)
# y = np.mean(losses, axis=1)
# err = np.stdev(losses, axis=1)
# plt.clf()
# plt.title("Post-selected mean losses")
# plt.xlabel("Block-depth")
# plt.ylabel("QKL")
# plt.bar(x, y, color='darkblue')
# plt.errorbar(x, y, yerr=err, fmt='o', color='gray')
# plt.savefig(OUTDIR + "circcomp.pdf")
