from distutils.log import error
from ___constants import (
    PARAM_PER_QUBIT_PER_DEPTH, 
    Q_MODE_GD, Q_MODE_NM, Q_MODE_ADAM, 
    Q_MODE_G_DIRECT_L, Q_MODE_AQGD, Q_MODE_CG,
    Q_MODE_G_ESCH, Q_MODE_G_ISRES, Q_MODE_NFT, 
    Q_MODE_SPSA, Q_MODE_TNC, 
    Q_MODES,
    NOISE_OPS
)
from __loss_funcs import KL
from __helpers import param_to_unitary
from __class_CNet import CNet
from __class_PQC import PQC
import numpy as np
import numpy.random as npr
from qiskit.algorithms.optimizers import (
    ADAM, AQGD, CG, GradientDescent,
    NELDER_MEAD, NFT, SPSA, TNC, 
    ESCH, ISRES, DIRECT_L
)
import torch as t
from math import pi


"""
HQNet: Hybrid quantum network. This is the full scheme for learning the symmetries
of a given quantum state. It combines a parametrized quantum circuit (PQC)
and a classical deep network (CNet), which learns to estimate the loss within 
the HQNet to regularize against finding known symmetries.

Currently built to perform a number of local optimizations and global search algorithms.
The choice is specified in `mode`, and the options are given in an TypeError upon
specifying an invalid option.

If `estimate` is True, then the PQC evaluates metrics using a sampling-based estimate
of the distribution rather than the true one, which cannot be calculated in real
quantum hardware. The units are exponential or polynomial in `L`, with noise scale `s_eps`.

The `noise` parameter accepts an integer from 0 to 2 and is to be used in classical simulation
on the qiskit backend. (On quantum hardware, ensure that `noise = 0`; quantum hardwawre will
be inevitably noisy and doesn't require simulation of such noise.) By default, `noise = 0`, 
meaning that the PQC will be noiseless in the classical simulation. If `noise = 1`, there will
be a small amount of depolarizing errors on single-qubit gates. If `noise = 2`, there will also
be more general unitary errors on all of the gates. 
"""

class HQNet:
    def __init__(self, state, bases, eta=1e-2, maxiter=1000,
                 metric_func=KL, mode=Q_MODE_NM, regularize=False, disp=False,
                 reg_scale=3, depth=0, estimate=False, poly=None, s_eps=50,
                 noise=0, state_prep_circ=None, error_prob=0.01, ops=None,
                 sample=False, jump=False, checkpoint=300
                 ):
        """
        Use a quantumfication of loss metric `metric_func` 
        over each basis in the list of bases `bases`.
        
        If regularize is True, `reg_scale` should be set to a number on the same order 
        as that of a poorly optimized quantum loss metric (e.g. KL divergence). 
        This requires numerical analysis on a system-by-system basis.
        """
        assert mode in Q_MODES, f"Mode {mode} not one of valid choices {Q_MODES}"
        assert noise in NOISE_OPS, f"Invalid noise parameter {noise}, must be in {NOISE_OPS}"
        if noise > 0:
            assert state_prep_circ is not None, "Must give a state preparation quantum circuit for noisy circuit simulation"
        
        maxiter = self.maxiter = int(maxiter)
        self.checkpoint = checkpoint
        if jump:
            maxiter = checkpoint
        self.regularize = regularize
        self.jump = jump
        assert not(regularize and jump), "Cannot regularize and jump: choose one"
        self.mode = mode
        self.depth = depth
        self.L = state.num_qubits
        self.CNets = [CNet(
                            self.L, depth=self.depth
                          ) 
                      for _ in bases]
        self.PQCs = [PQC(
                        state, 
                        basis_param=basis, 
                        metric_func=metric_func,
                        depth=self.depth,
                        estimate=estimate,
                        nrun=s_eps,
                        noise=noise,
                        state_prep_circ=state_prep_circ,
                        error_prob=error_prob,
                        poly=poly,
                        say_hi=False, 
                        ops=ops, 
                        sample=sample
                        )
                for basis in bases]
        self.regloss = lambda x: reg_scale * (2 * (1/(1 + 2*np.exp(-1/(2.5*np.float_power((x[0]-x[1])**2, 1/10)))) - 1/2))**2
        self.qloss = lambda x: x[0] + self.regloss(x)
        self.num_bases = len(bases)
        self.n_param = (self.depth + 1) * PARAM_PER_QUBIT_PER_DEPTH * self.L
        
        # Choose an algorithm including local (no predix)
        # and global (prefixed with g-) search algorithms on qiskit.
        if self.mode == Q_MODE_ADAM:
            self.optimizer = ADAM(lr=eta)
        elif self.mode == Q_MODE_GD:
            self.optimizer = GradientDescent(maxiter=maxiter, learning_rate=eta)
        elif self.mode == Q_MODE_NM:
            self.optimizer = NELDER_MEAD(maxiter=maxiter, maxfev=maxiter, adaptive=True, disp=disp)
        elif self.mode == Q_MODE_CG:
            self.optimizer = CG(maxiter=maxiter, disp=disp)
        elif self.mode == Q_MODE_AQGD:
            self.optimizer = AQGD(eta=eta)
        elif self.mode == Q_MODE_NFT:
            self.optimizer = NFT(disp=disp, maxfev=maxiter)
        elif self.mode == Q_MODE_SPSA:
            self.optimizer = SPSA(maxiter=maxiter)
        elif self.mode == Q_MODE_TNC:
            self.optimizer = TNC(maxiter=maxiter, disp=disp)
        elif self.mode == Q_MODE_G_ESCH:
            self.optimizer = ESCH(max_evals=maxiter)
        elif self.mode == Q_MODE_G_ISRES:
            self.optimizer = ISRES(max_evals=maxiter)
        elif self.mode == Q_MODE_G_DIRECT_L:
            self.optimizer = DIRECT_L(max_evals=maxiter)
        else:
            raise TypeError(f'Invalid choice of algorithm: {self.mode}')
        if disp:
            print(f"{self.L}-qubit (noise: {noise}) {'non-' if not regularize else ''}regularized '{self.mode}' hybrid quantum net initialized -- Hello world!")
    
    def __quantum_loss_metric(self, classical_loss_tensor):
        """
        The classical loss tensor is of order 3: (num bases, 2)
        where the last index are pairs (true metric, estimated metric). The 
        quantum loss metric takes a positive monotonic map over the estimated metric,
        then takes its difference with the true metric. It then sums over the bases.
        
        If not regularizing, then the loss is simply the KL divergence.
        """
        if self.regularize:
            return np.sum(np.apply_along_axis(self.qloss, 1, classical_loss_tensor.numpy()), 0)
        else:
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
        
        If regularization is used then `cnet.run_then_enq(p)` will build a queue of 
        data points being explored in the present search. Once this search epoch completes,
        the queue is trained together as a batch to upgrade the knowledge of the CNet, making
        it an adaptive regularizer. If the regularizer is not used, then the second piece of
        the classical loss tensor is discarded.
        
        Note: to train the CNet we will need a concatenation of the parameter and the 
        true metric. See `quantum_loss_metric()` for documentation on `classical_loss_tensor`.
        """
        p_vec = p_vec if t.is_tensor(p_vec) else t.from_numpy(p_vec)
        p_tens = t.zeros((self.num_bases, p_vec.size()[0] + 1))
        for i in range(self.num_bases):
            p_tens[i,:-1] = p_vec.clone()
        classical_loss_tensor = t.zeros((self.num_bases, 2))
        classical_loss_tensor[:,0] = t.tensor([qc.evaluate_true_metric(p_vec) for qc in self.PQCs])
        p_tens[:,-1] = classical_loss_tensor[:,0]
        classical_loss_tensor[:,1] = t.tensor([cnet.run_then_enq(p)
                                                for p, cnet in zip(p_tens, self.CNets)]) # estimated metric
        # print(classical_loss_tensor)
        # print(f"Loss = {self.__quantum_loss_metric(classical_loss_tensor)}")
        
        return classical_loss_tensor
    
    def param_to_quantum_loss(self, p_vec):
        classical_loss_tensor = self.get_classical_loss(p_vec)
        return self.__quantum_loss_metric(classical_loss_tensor)
    
    def param_to_jump_indicator(self, p_vec, thres=0.01):
        """Returns True if the CNet predicts it well enough"""
        classical_loss_tensor = self.get_classical_loss(p_vec).numpy()
        mse = (classical_loss_tensor[:,0]-classical_loss_tensor[:,1])**2
        # print(f"CLT = {classical_loss_tensor}")
        # print(f"Jump MSE = {mse}")
        return np.mean(mse) < thres
    
    def param_to_regval(self, p_vec):
        classical_loss_tensor = self.get_classical_loss(p_vec).numpy()
        mse = (classical_loss_tensor[:,0]-classical_loss_tensor[:,1])**2
        return mse
    
    def param_to_regloss(self, p_vec):
        assert self.regularize, "Must use regularizer to call this function"
        classical_loss_tensor = self.get_classical_loss(p_vec)
        print(f"True QKL\'s = {classical_loss_tensor[:,0]}")
        print(f"Predicted QKL\'s = {classical_loss_tensor[:,1]}")
        print(f"MSE = {np.square(classical_loss_tensor[:,0] - classical_loss_tensor[:,1])}")
        return np.sum(np.apply_along_axis(self.regloss, 1, classical_loss_tensor.numpy()), 0)
    
    def find_potential_symmetry(self, x0=None, print_log=True, 
                                reg_eta=1e-2, reg_nepoch=2000, include_nfev=False):
        """
        Run the optimization algorithm to obtain the maximally symmetric
        parameter, regularizing with the CNet. Train the CNet in parallel.
        Start at a random initial parameter `theta_0`.
        
        If `self.regularize = False`, then the `reg_*` parameters are ignored.
        
        In `algo_ops` one can specify specific options for the algorithm of choice. 
        In nelder-mead, `disp : Bool` and `adaptive : Bool` indicate whether to 
        show convergence messages and use the high-dimensional adaptive algorithm,
        respectively. In SGD, `num_epochs : Int` is the number of times we choose
        a random parameter and do SGD over, `h : Float` is the finite difference parameter.
        
        RETURNS: proposed symmetry
        """
        theta_0 = self.PQCs[0].gen_rand_data(1, include_metric=False).squeeze() \
            if x0 is None else t.tensor(x0)
        n_param = theta_0.shape[0]
        bounds = [(0, 2*pi)] * n_param

        if self.jump:
            njump = 0
            for _ in range(self.maxiter // self.checkpoint):
                point, value, nfev = self.optimizer.optimize(n_param, 
                                                            self.param_to_quantum_loss, 
                                                            initial_point=theta_0, 
                                                            variable_bounds=bounds)
                if nfev < self.checkpoint:
                    break # optimization done
                if self.param_to_jump_indicator(point):
                    theta_0 = self.PQCs[0].gen_rand_data(1, include_metric=False).squeeze()
                    njump += 1
                    unitary = param_to_unitary(theta_0.numpy().reshape((self.L, 
                                                                        PARAM_PER_QUBIT_PER_DEPTH)))
            if print_log:
                print(f"Jumped {njump} times")
        else:
            point, value, nfev = self.optimizer.optimize(n_param, 
                                                        self.param_to_quantum_loss, 
                                                        initial_point=theta_0, 
                                                        variable_bounds=bounds)
        
        if print_log:
            print(f"Optimized to loss metric = {value}")
            print(f"Queried loss func {nfev} times")

        regularizer_losses = None
        if self.regularize or self.jump:
            [cnet.flush_q(nepoch=reg_nepoch, 
                          eta=reg_eta) for cnet in self.CNets]
            regularizer_losses = self.param_to_regloss(point) if self.regularize \
                                                else self.param_to_regval(point)
        else:
            [cnet.kill_q() for cnet in self.CNets]
        if include_nfev:
            return point, value, nfev
        return point, value, regularizer_losses
        
        
        
        
    
    