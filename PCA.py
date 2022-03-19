"""
Explicitly display the symmetries of the 3-GHZ state 
using dimensionality reduction.
"""
from __helpers import prepare_basis, param_to_unitary
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __loss_funcs import KL
from __class_BasisTransformer import BasisTransformer
from __class_PQC import PQC
from __class_HQNet import HQNet
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import argparse
from qiskit.quantum_info import Statevector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from GHZ_generator import GHZ_state_circuit

# Identify each symmetry Type I (diag) or Type II (off-diag)
def classify_sym(unitary_product):
    type_1_points = 0
    type_2_points = 0
    for unitary in unitary_product:
        dsum = unitary[0,0] + unitary[1,1] 
        odsum = unitary[0,1] + unitary[1,0]
        if odsum == 0:
            type_1_points += 1
            continue
        elif dsum == 0:
            type_2_points += 1
            continue
        elif dsum / odsum > 100:
            type_1_points += 1
            continue
        elif odsum / dsum > 100:
            type_2_points += 1
            continue
    
    if type_1_points == NUM_QUBITS:
        return 1 # Certainly Type I
    elif type_2_points == NUM_QUBITS:
        return 2 # Certainly Type II
    elif type_1_points > type_2_points:
        return 3 # Probably Type I
    elif type_2_points > type_1_points:
        return 4 # Probably Type II
    else:
        return 5 # No idea
    
def type_to_color(type):
    if type == 1:
        return 'darkblue'
    if type == 2:
        return 'darkred'
    if type == 3:
        return 'cornflowerblue'
    if type == 4:
        return 'tomato'
    if type == 5:
        return 'gray'
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Show symmetries of 3-GHZ in 2D DR space")
    parser.add_argument("-d", "--depth", type=int, help='circuit block depth', default=0)
    parser.add_argument("-L", "--L", type=int, help='number of qubits', default=3)
    parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
    parser.add_argument("-r", "--reg", action='store_true', help='use regularizer')
    parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
    parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
    parser.add_argument("nrun", type=int, help='number of symmetries to find')
    args = parser.parse_args()

    def dprint(msg):
        if args.verbose:
            print(msg)

    CIRCUIT_DEPTH = args.depth
    NUM_QUBITS = args.L
    NUM_BASES = args.bases
    USE_REGULARIZER = args.reg
    LOSS_METRIC = KL
    ESTIMATE = False
    SAMPLE = False
    NOISE_SCALE = 5
    NRUN = args.nrun
    OUTDIR = (args.out + '/') if args.out[-1] != '/' else args.out
    OPS = None # MMD sigma parameter

    # Prepare state noiselessly
    state = Statevector.from_int(0, 2**NUM_QUBITS)
    qc = GHZ_state_circuit(L=NUM_QUBITS)
    state = state.evolve(qc)
    dprint("State preparation circuit:\n" + qc)

    # Start the HQNet
    bases = prepare_basis(state.num_qubits, num=NUM_BASES)
    hqn = HQNet(state, bases, eta=1e-2, maxiter=1E4, disp=False,
                mode='Nelder-Mead', depth=CIRCUIT_DEPTH, 
                estimate=ESTIMATE, s_eps=NOISE_SCALE, 
                metric_func=LOSS_METRIC, ops=OPS, sample=SAMPLE, 
                jump=USE_REGULARIZER)
    dprint("Variational circuit:")
    if args.verbose:
        hqn.view_circuit().draw()

    # Find symmetries
    param_shape = (state.num_qubits, CIRCUIT_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)
    param_dim = np.prod(param_shape)
    proposed_syms = t.zeros((NRUN, param_dim))

    losses = np.zeros(NRUN)
    for i in range(NRUN):
        potential_sym, losses[i], _ = hqn.find_potential_symmetry(print_log=args.verbose)
        proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
        potential_sym = potential_sym.reshape(param_shape)
    print(f"\nMedian loss: {np.median(losses)}, stdev: {np.std(losses)}")
    np.save(OUTDIR + 'losses.npy', losses)

    # Pushforward from parameter space to SU(2)^((x) 3)
    unitaries_prods = np.array([np.around(
                            param_to_unitary(
                                sym.reshape((NUM_QUBITS, -1)).numpy()), 4) 
                        for sym in proposed_syms])
    np.save(OUTDIR + 'unitaries.npy', unitaries_prods)
        
    sym_labels = np.array([type_to_color(classify_sym(unitary_prod)) \
                            for unitary_prod in unitaries_prods])

    # Project from the Lie group to 2D space
    unitary_vecs = StandardScaler().fit_transform(unitaries_prods.reshape((NRUN, -1)))
    pca = PCA(n_components=2)
    fit = pca.fit_transform(unitary_vecs)
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio = {explained_variance}")

    # Save the data and plot it
    np.save(OUTDIR + 'syms_2D.npy', fit)
    np.save(OUTDIR + 'syms_2D_labs.npy', sym_labels)

    plt.clf()
    plt.xlabel(r'PC$_1$')
    plt.ylabel(r'PC$_2$')
    plt.scatter(fit[:,0], fit[:,1], sym_labels)
    plt.savefig(OUTDIR + 'syms_2D.pdf')
    plt.show()
