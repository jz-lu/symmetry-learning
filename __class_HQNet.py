from inspect import Parameter
from ___constants import (
    Q_MODE_GD, Q_MODE_NM, Q_MODE_ADAM, 
    Q_MODE_G_DIRECT_L, Q_MODE_AQGD, Q_MODE_CG,
    Q_MODE_G_ESCH, Q_MODE_G_ISRES, Q_MODE_NFT, 
    Q_MODE_SPSA, Q_MODE_TNC, 
    Q_MODES, 
    DEFAULT_QNET_OPS, 
    MINIMUM_LR
)
from __loss_funcs import KL
from __class_CNet import CNet
from __class_PQC import PQC
import numpy as np
import numpy.random as npr
from qiskit.algorithms.optimizers import (
    ADAM, AQGD, CG, GradientDescent,
    NELDER_MEAD, NFT, SPSA, TNC, 
    ESCH, ISRES, DIRECT_L
)
from scipy.optimize import minimize
import torch as t
from math import pi


"""
HQNet: Hybrid quantum network. This is the full scheme for learning the symmetries
of a given quantum state. It combines a parametrized quantum circuit (PQC)
and a classical deep network (CNet), which learns to estimate the loss within 
the HQNet to regularize against finding known symmetries.

Currently built to perform either gradient descent via stochastic finite differences 
(mode=gd) or Nelder-Mead simplex search (mode=nm).
"""

class HQNet:
    def __init__(self, state, bases, eta=1e-2, maxiter=1000,
                 metric_func=KL, mode=Q_MODE_ADAM, regularize=False, disp=False,
                 reg_scale=3):
        """
        Use a quantumfication of loss metric `metric_func` 
        over each basis in the list of bases `bases`.
        
        If regularize is True, `reg_scale` should be set to a number on the same order 
        as that of a poorly optimized quantum loss metric (e.g. KL divergence). 
        This requires numerical analysis on a system-by-system basis.
        """
        assert mode in Q_MODES, f"Mode {mode} not one of valid choices {Q_MODES}"
        self.regularize = regularize
        self.mode = mode
        self.L = state.num_qubits
        self.CNets = [CNet(
                            self.L
                          ) 
                      for _ in bases]
        self.PQCs = [PQC(
                        state, 
                        basis_param=basis, 
                        metric_func=metric_func
                        )
                for basis in bases]
        self.qloss = lambda x: x[0] + reg_scale * np.tanh((x[0]-x[1])**2)
        self.num_bases = len(bases)
        
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
            self.optimizer = NFT(disp=disp)
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
        
        print(f"{self.L}-qubit {'non-' if not regularize else ''}regularized '{self.mode}' hybrid quantum net initialized -- Hello world!")
    
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
        
    def param_to_quantum_loss(self, p_vec):
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
        classical_loss_tensor[:,1] = t.tensor([cnet.run_then_enq(p)
                                                for p, cnet in zip(p_tens, self.CNets)]) # estimated metric
        
        return self.__quantum_loss_metric(classical_loss_tensor)
    
    def find_potential_symmetry(self, print_log=True, reg_eta=1e-2, reg_nepoch=2000):
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
        
        ? Open question: will the CNet performance get worse when the batch is 
        ? a local path in the space, not a randomly sampled set of parameters?
        ? (We want it to still be able to interpolate very well, but we DON'T 
        ? want it to extrapolate well. That's the point of regularizing.)
        """
        theta_0 = self.PQCs[0].gen_rand_data(1, include_metric=False).squeeze()
        n_param = theta_0.shape[0]
        bounds = [(0, 2*pi)] * n_param

        point, value, nfev = self.optimizer.optimize(n_param, 
                                                     self.param_to_quantum_loss, 
                                                     initial_point=theta_0, 
                                                     variable_bounds=bounds)

        if self.regularize:
            regularizer_losses = [cnet.flush_q() for cnet in self.CNets]
        
        if print_log:
            print(f"Optimized to QKL = {value}")
            print(f"Queried loss func {nfev} times")
        return point, value, regularizer_losses
        
        
        
        
    
    