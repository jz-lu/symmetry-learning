#* Uncomment these two lines if running directly on local MacOS. It has some weird OS problem that this line magically fixes.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
"""

from math import pi
from __class_PQC import PQC
from __class_HQNet import HQNet
import numpy as np
import torch as t
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, QuantumRegister
import argparse
from qiskit.algorithms.optimizers import (
    ADAM, AQGD, CG, GradientDescent,
    NELDER_MEAD, NFT, SPSA, TNC, 
    ESCH, ISRES, DIRECT_L
)

parser = argparse.ArgumentParser(description="Show symmetries of 3-GHZ in 2D DR space")
parser.add_argument("-d", "--depth", type=int, help='circuit block depth', default=1)
parser.add_argument("-L", "--L", type=int, help='number of qubits', default=3)
parser.add_argument("-b", "--bases", type=int, help='number of bases', default=2)
parser.add_argument("-r", "--reg", action='store_true', help='use regularizer')
parser.add_argument("-o", "--out", type=str, help='output directory', default='.')
parser.add_argument("-v", "--verbose", action='store_true', help='display outputs')
parser.add_argument("-s", "--scale", type=int, help='number of samplings', default=50)
parser.add_argument("nrun", type=int, help='number of symmetries to find')
args = parser.parse_args()

def dprint(msg):
    if args.verbose:
        print(msg)

BLOCK_DEPTH = args.depth
SAMPLING_DENSITY = 200
NUM_QUBITS = args.L
NUM_BASES = args.bases
USE_REGULARIZER = args.reg
NRUN = args.nrun
OUTDIR = (args.out + '/') if args.out[-1] != '/' else args.out
PARAM_PER_QUBIT_PER_DEPTH = 3
NOISE_SCALE = args.scale

def GHZ_state_circuit(L=3, qreg=None):
    # Create GHZ state circuit
    GHZ_circ = QuantumCircuit(QuantumRegister(L) if qreg is None else qreg)
    GHZ_circ.h(0)
    for i in range(1, L):
        GHZ_circ.cx(0, i)
    return GHZ_circ

def qubit_expansion(L):
    """Returns list of all L-qubits in z basis in lexicographical order"""
    assert isinstance(L, int) and L > 0
    d2b = np.vectorize(np.binary_repr)
    return d2b(np.arange(2**L), L)

def qubit_retraction(bitstr):
    """Inverse operation to expansion: given bitstring, return decimal"""
    return int(bitstr, 2)

def prepare_basis(L, num=2, init=0):
    init_basis = t.zeros(L, 3) # z-basis
    init_basis[:,0] += init * t.ones(L)
    bases = [init_basis]
    basis_here = init_basis
    for _ in range(num-1):
        perturbation = t.zeros(L, 3)
        perturbation[:,0] += pi/10 * t.ones(L) # small rotation in one axis for each qubit
        basis_here = basis_here + perturbation
        bases.append(basis_here)
    return bases

def rand_basis(L):
    """Returns a random basis choice parametrized by (theta, phi, lambda) for L qubits"""
    return (2 * pi * t.rand(3)).repeat((L, 1))

def KL(P, Q, eps=1e-7):
    """KL divergence on two discrete distributions"""
    P = P + eps * np.ones_like(P)
    Q = Q + eps * np.ones_like(Q)
    return np.sum(P * np.log(P/Q))

class BasisTransformer:
    def __init__(self, states, updated_parameters):
        self.states = states
        self.L = states[0].num_qubits
        assert np.all([states[0].num_qubits == state.num_qubits for state in states])
        if type(updated_parameters).__module__ == t.__name__:
            self.p = updated_parameters.clone().cpu().detach().numpy()
        else:
            assert type(updated_parameters).__module__ == np.__name__
            self.p = updated_parameters
        self.p = -self.p # *= -1 since measurement ~= inverted rotation
        self.__transform()
    
    def __make_qc(self):
        """
        Make the quantum circuit in qiskit to execute the transformation. Rotation
        performed with qiskit U-gates: https://qiskit.org/documentation/stubs/qiskit.circuit.library.UGate.html
        
        param_mat: L x L matrix of (theta, phi, lambda) parameters (a triple for each row)
        """
        qubits = QuantumRegister(self.L)
        qc = QuantumCircuit(qubits)
        
        for i in range(self.L):
            qc.u(*self.p[i], qubits[i])
        self.qc = qc
    
    def __transform(self):
        """Perform the change of basis for each state"""
        self.__make_qc()
        self.transformed_states = [state.copy().evolve(self.qc) for state in self.states]
    
    def transformed_states(self):
        return self.transformed_states
    
    def updated_dist(self, poly=None, nrun=100, sample=False):
        return [state.probabilities() for state in self.transformed_states]

class PQC:
    def __init__(self, state, basis_param=None, 
                 metric_func=KL, depth=0, 
                 nrun=50, 
                 state_prep_circ=None, qreg=None,
                 poly=None, say_hi=True, ops=None):
        assert depth >= 0 and isinstance(depth, int), f"Invalid circuit depth {depth}"
        self.nrun = nrun
        self.metric = metric_func
        self.state = state
        self.L = state.num_qubits
        self.depth = depth
        self.poly = poly
        self.qreg = qreg
        self.state_prep_circ = state_prep_circ
        self.n_param = self.L * PARAM_PER_QUBIT_PER_DEPTH * (self.depth + 1)
        self.ops = ops
            
        if type(basis_param).__module__ == t.__name__:
            self.bp = basis_param.clone().cpu().detach().numpy()
        elif type(basis_param).__module__ == np.__name__:
            self.bp = basis_param
        self.bp = -self.bp # *= -1 since measurement ~= inverted rotation
        
        # Obtain distribution of state when measured in the given basis.
        if basis_param is not None:
            self.basis_dist = BasisTransformer([state], basis_param).updated_dist(
                poly=poly, nrun=nrun
                )[0]
        else:
            self.basis_dist = state.probabilities()
        
        # Choose the appropriate quantum circuit based on noise level
        self.evolver = self.__Q_th

        if say_hi:
            print(f"Parametrized quantum circuit ({'' if ops is None else ', ops=%d'%ops}) initialized.")
    
    def __make_Q_th(self, p, Q_th=None, noisy=False):
        """
        Make a quantum circuit in qiskit corresponding to parameters p. Currently, 
        the model used is a linear entanglement circuit with depth `self.depth`. This
        uses CNOT gates to entangle nearest neighbors `self.depth` times. Each sequence
        of entangling CNOT gates is sandwiched by universal single-qubit rotations.
        
        The most universal circuit of a given depth is a full entanglement, over every pair.
        We will restrict ourselves to a local version, for the time being at least.
        """
        qubits = QuantumRegister(self.L)
        Q_th = QuantumCircuit(qubits)
        if noisy: # GHZ only
            Q_th.h(0)
            for i in range(1, self.L):
                Q_th.cx(0, i)
        assert p.shape[0] == self.n_param, f"Expected param shape {self.n_param}, got {p.shape[0]}"
        p = t.reshape(p, (self.L, self.depth+1, PARAM_PER_QUBIT_PER_DEPTH))
        if type(p).__module__ == t.__name__:
            p = p.clone().cpu().detach().numpy()
        
        # Quantum parametrization -- DYNAMIC (change based on circuit param)
        for i in range(self.L):
            Q_th.ry(p[i,0,0], qubits[i])
            Q_th.rz(p[i,0,1], qubits[i])
        for d in range(1, self.depth + 1):
            for i in range(self.L-1):
                Q_th.cx(i, i+1)
            for i in range(self.L):
                Q_th.ry(p[i,d,0], qubits[i])
                Q_th.rz(p[i,d,1], qubits[i])
        
        # Measurement in the basis -- STATIC (do not change regardless of parameterization)
        for i in range(self.L):
            Q_th.u(*self.bp[i], qubits[i])
            
        return Q_th
    
    def __Q_th(self, p):
        """
        Apply the quantum circuit in qiskit corresponding to Q_theta with parameters p.
        """
        Q_th = self.__make_Q_th(p)
        return self.state.copy().evolve(Q_th)
    
    def get_circ(self, p):
        return self.__make_Q_th(p)
        
    def evaluate_true_metric(self, p):
        """
        Calculate the metric loss of Q_th(p)|state> against reference |state>,
        where p parametrizes the circuit Q_th.
        
        If the PQC is on estimation mode, the metric is computed on a sampling-based
        estimate of the distribution. Unavoidably, the sampling has exponential
        time complexity in L. Otherwise, the qiskit calculated backend distribution is
        used, but the latter is incompatible with calculations on true quantum hardware.
        
        ? Open question: If the calculation is exponential anyway, we cannot get an 
        ? advantage via quantum computers. Thus, is this problem fundamentally limited
        ? to classical calculation? Unless, we can somehow learn with a polynomial sample...
        ? Alternatively, we can define the "approximate symmetry learning problem", which works
        ? to identify symmetries that have a distribution well-approximated by a small subset.
        ? Or, more wildly, is it possible to do this with polynomial sampling???
        """
        state = self.evolver(p)
        distribution = state.probabilities() #! Need to change on hardware?
        if self.ops is None:
            return self.metric(self.basis_dist, distribution)
        else:
            return self.metric(self.basis_dist, distribution, self.ops)
    
    def true_metric_on_params(self, p_list):
        """Same as `evaluate_true_metric` but over a list of parameters"""
        assert len(p_list) > 0, "Parameter list empty"
        return [self.evaluate_true_metric(p) for p in p_list]
    
    def gen_rand_data(self, sz, include_metric=True):
        """
        Generate `sz` random input parameters (just one axis rotation for now), 
        which inputs into the CNet, 
        sampled from X ~ 2 * Pi * DUnif(n), whwere n = SAMPLING_DENSITY.
        """
        if include_metric:
            dataset = t.zeros(sz, self.n_param + 1)
            params = t.randint(0, SAMPLING_DENSITY, (sz, self.n_param)) * 2 * pi / SAMPLING_DENSITY
            dataset[:,:-1] = params # the last column of the dataset stores the output loss metric
            true_metric = t.tensor([self.evaluate_true_metric(param) for param in params]) # target value for CNet
            dataset[:,-1] = true_metric
        else:
            dataset = t.randint(0, SAMPLING_DENSITY, (sz, self.n_param)) * 2 * pi / SAMPLING_DENSITY
        return dataset

class HQNet:
    def __init__(self, state, bases, eta=1e-2, maxiter=1000,
                 metric_func=KL, disp=False,
                 depth=0, poly=None, s_eps=50,
                 ops=None
                 ):
        """
        Use a quantumfication of loss metric `metric_func` 
        over each basis in the list of bases `bases`.
        """
        maxiter = self.maxiter = int(maxiter)
        self.depth = depth
        self.L = state.num_qubits
        self.PQCs = [PQC(
                        state, 
                        basis_param=basis, 
                        metric_func=metric_func,
                        depth=self.depth,
                        nrun=s_eps,
                        poly=poly,
                        say_hi=False, 
                        ops=ops, 
                        )
                for basis in bases]
        self.qloss = lambda x: x[0]
        self.num_bases = len(bases)
        self.n_param = (self.depth + 1) * PARAM_PER_QUBIT_PER_DEPTH * self.L
        
        # Choose an algorithm including local (no predix)
        # and global (prefixed with g-) search algorithms on qiskit.
        self.optimizer = NELDER_MEAD(maxiter=maxiter, maxfev=maxiter, adaptive=True, disp=disp) #* change if you want
        if disp:
            print(f"{self.L}-qubit hybrid quantum net initialized -- Hello world!")
    
    def __quantum_loss_metric(self, classical_loss_tensor):
        """
        The KL divergence.
        """
        return t.sum(classical_loss_tensor[:,0]).item()
    
    def view_circuit(self):
        """
        Give a circuit with some random parameters. The purpose of the function is to let
        the user draw the circuit and check that the architecture looks right, not to 
        determine if the parameters used are the right ones.
        """
        return self.PQCs[0].get_circ(t.zeros(self.n_param))
        
    def get_classical_loss(self, p_vec):
        """
        Function mapping a parameter to the quantum loss.
        `p_vec` is a vectorized parametrization (size: dim theta) of the PQC.

        Ignore column 2 of the tensor, since it is an artifact of regularization code.
        
        Note: to train the CNet we will need a concatenation of the parameter and the 
        true metric. See `quantum_loss_metric()` for documentation on `classical_loss_tensor`.
        """
        p_vec = p_vec if t.is_tensor(p_vec) else t.from_numpy(p_vec)
        p_tens = t.zeros((self.num_bases, p_vec.size()[0] + 1))
        for i in range(self.num_bases):
            p_tens[i,:-1] = p_vec.clone()
        classical_loss_tensor = t.zeros((self.num_bases, 2))
        classical_loss_tensor[:,0] = t.tensor([qc.evaluate_true_metric(p_vec) for qc in self.PQCs])
        return classical_loss_tensor
    
    def param_to_quantum_loss(self, p_vec):
        classical_loss_tensor = self.get_classical_loss(p_vec)
        return self.__quantum_loss_metric(classical_loss_tensor)
    
    def find_potential_symmetry(self, x0=None, print_log=True, include_nfev=False):
        """
        Run the optimization algorithm to obtain the maximally symmetric
        parameter, without regularization.

        In `algo_ops` one can specify specific options for the algorithm of choice. 
        In nelder-mead, `disp : Bool` and `adaptive : Bool` indicate whether to 
        show convergence messages and use the high-dimensional adaptive algorithm,
        respectively. In SGD, `num_epochs : Int` is the number of times we choose
        a random parameter and do SGD over, `h : Float` is the finite difference parameter.
        
        RETURNS: proposed symmetry.
        """
        theta_0 = self.PQCs[0].gen_rand_data(1, include_metric=False).squeeze() \
            if x0 is None else t.tensor(x0)
        n_param = theta_0.shape[0]
        bounds = [(0, 2*pi)] * n_param

        result = self.optimizer.minimize(self.param_to_quantum_loss, 
                                                     theta_0, 
                                                     bounds=bounds)
        point, value, nfev = result.x, result.fun, result.nfev
        
        if print_log:
            print(f"Optimized to loss metric = {value}", flush=True)
            print(f"Queried loss func {nfev} times", flush=True)

        if include_nfev:
            return point, value, nfev
        return point, value
        

# Prepare state
state = Statevector.from_int(0, 2**NUM_QUBITS)
qc = GHZ_state_circuit(L=NUM_QUBITS)
print(qc)
state = state.evolve(qc)
param_shape = (state.num_qubits, BLOCK_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)

# Prepare bases
bases = prepare_basis(state.num_qubits, num=NUM_BASES, init=0)
num_bases = len(bases)

hqn = HQNet(state, bases, eta=1e-2, maxiter=1E4, depth=BLOCK_DEPTH, 
            s_eps=NOISE_SCALE)

param_dim = np.prod(param_shape)
proposed_syms = t.zeros((NRUN, param_dim))
losses = np.zeros(NRUN)
avg = 0
for i in range(NRUN):
    print(f"== Run {i+1}/{NRUN} ==")
    potential_sym, losses[i] = hqn.find_potential_symmetry(print_log=True)
    proposed_syms[i] = potential_sym if t.is_tensor(potential_sym) else t.from_numpy(potential_sym)
    potential_sym = potential_sym.reshape(param_shape)
    print(f"Proposed symmetry:\n{potential_sym}\n")
print(f"\nAverage loss: {np.mean(losses)}")

# Save the outputs
np.save(f"{OUTDIR}syms_hardware.npy", proposed_syms)
np.save(f"{OUTDIR}losses_hardware.npy", losses)