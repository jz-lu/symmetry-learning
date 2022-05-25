from ___constants import (
    CNET_TEST_SIZE, CNET_TRAIN_SIZE, SAMPLING_DENSITY, 
    PARAM_PER_QUBIT_PER_DEPTH,
    NOISE_OPS
)
from __loss_funcs import KL
from __helpers import qubit_retraction, qubit_expansion
from __class_BasisTransformer import BasisTransformer
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.providers.aer.noise import pauli_error, depolarizing_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit.tools.visualization import plot_histogram
import numpy as np
import torch as t
from math import pi

"""
PQC: parametrized quantum circuit. Having chosen (in advance) a parametrization 
for a family of Q-circuits, we map the parameter to its corresponding circuit below. 

`basis_param` is a L x depth+1 x PARAM_PER_QUBIT_PER_DEPTH matrix of parameters 
specifying a tensor product of rotations in the L-tensored Bloch sphere. 
The exact parametrization is specified by Qiskit's Ry and Rz gates.

Application of the circuit on a set of quantum states generates a dataset of states
evolved by the PQC. 

This class is static in the HQNet scheme, in the sense that we perform optimization on the 
parameter space, not the circuit architecture itself. That is, the map param -> circuit
is fixed; we simply want to find the parameter `param*` that produces a circuit leaving the
desired pristine state invariant, thereby finding a symmetry of the pristine state.

`metric_func`: choice of loss metric. Many others are available in `__loss_funcs.py`. 
Defaults to KL divergence.

If `estimate` is True, the loss metric at each basis will be estimated by 
creating `nrun * 2^L` copies of the state, running them through the circuit, and then creating
an empirical distribution. If `estimate` is False, then the true distribution stored
in the Qiskit backend will be used instead.

If `metric_samples` is True, then the PQC assumes that the supplied metric function is a 
sampling-based metric and will supply it with samples instead of a distribution. An example
if the empirical MMD loss.
"""

class PQC:
    def __init__(self, state, basis_param=None, 
                 metric_func=KL, depth=0, 
                 estimate=False, nrun=50, 
                 noise=0, markovian=False, state_prep_circ=None, qreg=None, error_prob=0.001, 
                 poly=None, say_hi=True, ops=None, sample=False):
        assert noise in NOISE_OPS, f"Invalid noise parameter {noise}, must be in {NOISE_OPS}"
        if noise > 0:
            assert state_prep_circ is not None, "Must give a state preparation quantum circuit for noisy circuit simulation"
            
        assert depth >= 0 and isinstance(depth, int), f"Invalid circuit depth {depth}"
        self.estimate = estimate
        self.nrun = nrun
        self.metric = metric_func
        self.state = state
        self.L = state.num_qubits
        self.depth = depth
        self.poly = poly
        self.qreg = qreg
        self.state_prep_circ = state_prep_circ
        self.markovian = markovian
        self.error_prob = error_prob
        self.n_param = self.L * PARAM_PER_QUBIT_PER_DEPTH * (self.depth + 1)
        self.ops = ops
        self.sample = sample
            
        if type(basis_param).__module__ == t.__name__:
            self.bp = basis_param.clone().cpu().detach().numpy()
        elif type(basis_param).__module__ == np.__name__:
            self.bp = basis_param
        self.bp = -self.bp # *= -1 since measurement ~= inverted rotation
        
        # Obtain distribution of state when measured in the given basis.
        if basis_param is not None:
            self.basis_dist = BasisTransformer([state], basis_param).updated_dist(
                    estimate=estimate, poly=poly, nrun=nrun, sample=sample
                )[0]
        else:
            if estimate and sample:
                self.basis_dist = np.array([qubit_retraction(state.measure()[0]) for _ \
                                        in range(2**self.L if self.poly is None else self.L**self.poly)])
            elif estimate:
                units = 2**self.L if self.poly is None else self.L**self.poly
                estimate = np.zeros(2**self.L)
                for _ in range(self.nrun * units):
                    estimate[qubit_retraction(state.measure()[0])] += 1
                self.basis_dist = estimate / (self.nrun * units)
            else:
                self.basis_dist = state.probabilities()
        
        # Choose the appropriate quantum circuit based on noise level
        self.evolver = self.__Q_th
        if noise == 1:
            self.evolver = self.__Q_th_noise_1
        elif noise == 2:
            self.evolver = self.__Q_th_noise_2
        
        if say_hi:
            print(f"Parametrized quantum circuit (noise: {noise}{'' if ops is None else ', ops=%d'%ops}) initialized.")
    
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
    
    def get_Q_th(self, p):
        """
        Get quantum circuit associated with parameter `p`.
        """
        return self.__make_Q_th(p)

    def __Q_th_noise_1(self, p):
        """
        Apply the quantum circuit with small depolarizing noise.
        """
        Q_th = self.__make_Q_th(p, Q_th=self.state_prep_circ)
        error_depol = depolarizing_error(self.error_prob, self.L)
        noise_depol = NoiseModel()
        noise_depol.add_all_qubit_quantum_error(error_depol, "depolarize")
        sim_dpnoise = AerSimulator(noise_model=noise_depol)
        circ_dpnoise = transpile(Q_th, sim_dpnoise)
        result_dp = sim_dpnoise.run(circ_dpnoise).result()
        counts_dp = dict.fromkeys(qubit_expansion(self.L), 0)
        counts_dp.update(result_dp.get_counts(0))
        dp_dist = np.array([i[1] for i in sorted(counts_dp.items())]) / sum(counts_dp.values())
        return Statevector(dp_dist)

    def __Q_th_noise_2(self, p):
        """
        Apply the quantum circuit with general noise. If `self.markovian` is True,
        only reversible errors will be applied.
        """
        Q_th = self.__make_Q_th(p, Q_th=None, noisy=True)
        Q_th.measure_all()
        
        noise_bit_flip = NoiseModel()
        if not self.markovian:
            error_reset = pauli_error([('X', self.error_prob), ('I', 1 - self.error_prob)])
            error_meas = pauli_error([('X', self.error_prob), ('I', 1 - self.error_prob)])
            noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
            noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
        error_gate1 = pauli_error([('X', self.error_prob), ('I', 1 - self.error_prob)])
        error_gate2 = error_gate1.tensor(error_gate1)
        noise_bit_flip.add_all_qubit_quantum_error(error_gate1, 
                                                    ["u1", "u2", "u3", "rx", "ry", "rz"])
        noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
        sim_noise = None
        sim_noise = AerSimulator(noise_model=noise_bit_flip)
        
        circ_tnoise = transpile(Q_th, sim_noise)
        result_bit_flip = sim_noise.run(circ_tnoise).result()
        counts_bit_flip = dict.fromkeys(qubit_expansion(self.L), 0)
        # print(result_bit_flip.get_counts(0))
        counts_bit_flip.update(result_bit_flip.get_counts(0))
        bf_dist = np.array([i[1] for i in sorted(counts_bit_flip.items())]) / sum(counts_bit_flip.values())
        return Statevector(bf_dist)
    
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
        distribution = None
        if self.estimate and self.sample:
            distribution = np.array([qubit_retraction(state.measure()[0]) for _ \
                                        in range(2**self.L if self.poly is None else self.L**self.poly)])
        elif self.estimate:
            units = 2**self.L if self.poly is None else self.L**self.poly
            estimate = np.zeros(2**self.L)
            for _ in range(self.nrun * units):
                estimate[qubit_retraction(state.measure()[0])] += 1
            distribution = estimate / (self.nrun * units)
        else:
            distribution = state.probabilities()
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

    def generate_train_test(self, train_size=CNET_TRAIN_SIZE, test_size=CNET_TEST_SIZE):
        """
        Generate the train and test datasets for neural network training.
        A convenience function mostly for unit-testing the CNet.
        """
        train_data = self.gen_rand_data(train_size)
        test_data = self.gen_rand_data(test_size)
        return train_data, test_data