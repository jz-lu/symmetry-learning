from inspect import Parameter
from ___constants import (
    Q_MODE_GD, Q_MODE_NM, Q_MODES, 
    DEFAULT_QNET_OPS, 
    MINIMUM_LR
)
from __loss_funcs import KL
from __class_CNet import CNet
from __class_PQC import PQC
import numpy as np
import numpy.random as npr
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
    def __init__(self, state, bases, metric_func=KL, mode=Q_MODE_GD, regularize=False):
        """
        Use a quantumfication of loss metric `metric_func` 
        over each basis in the list of bases `bases`.
        """
        assert mode in Q_MODES, f"Mode {mode} not one of valid choices {Q_MODES}"
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
        self.regularizer_transform = lambda x: 0.5 * x # transformation to estimated metric
        self.qloss = lambda x: x[1] - self.regularizer_transform(x[0])
        self.num_bases = len(bases)
        self.regularize = regularize
        
        print(f"{'Non-r' if not regularize else 'R'}egularized hybrid quantum net initialized -- Hello world!")
    
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
        
        Note: to train the CNet we will need a concatenation of the parameter and the 
        true metric. See `quantum_loss_metric()` for documentation on `classical_loss_tensor`.
        """
        p_vec = t.from_numpy(p_vec) if self.mode == Q_MODE_NM else p_vec
        p_tens = t.zeros((self.num_bases, p_vec.size()[0] + 1))
        for i in range(self.num_bases):
            p_tens[i,:-1] = p_vec.clone()
        classical_loss_tensor = t.zeros((self.num_bases, 2))
        classical_loss_tensor[:,0] = t.tensor([qc.evaluate_true_metric(p_vec) for qc in self.PQCs])
        classical_loss_tensor[:,1] = t.tensor([cnet.run_then_train_SGD(p)[0]
                                                for p, cnet in zip(p_tens, self.CNets)]) # estimated metric
        return self.__quantum_loss_metric(classical_loss_tensor)
    
    def find_potential_symmetry(self, atol=1e-4, eta=1e-2, algo_ops=DEFAULT_QNET_OPS, print_log=False):
        """
        Run the optimization algorithm to obtain the maximally symmetric
        parameter, regularizing with the CNet. Train the CNet in parallel.
        Start at a random initial parameter `theta_0`.
                
        For singleton datum training only. We can use a Nelder-Mead simplex
        search or a finite difference stochastic gradient descent (FDSGD).
        
        In `algo_ops` one can specify specific options for the algorithm of choice. 
        In nelder-mead, `disp : Bool` and `adaptive : Bool` indicate whether to 
        show convergence messages and use the high-dimensional adaptive algorithm,
        respectively. In SGD, `num_epochs : Int` is the number of times we choose
        a random parameter and do SGD over, `h : Float` is the finite difference parameter.
        
        RETURNS: proposed symmetry, (stochastic parameter empirical distribution) <-- if mode is FDSGD
        
        ? Open question: will the CNet performance get worse when the batch is 
        ? a local path in the space, not a randomly sampled set of parameters?
        ? (We want it to still be able to interpolate very well, but we DON'T 
        ? want it to extrapolate well. That's the point of regularizing.)
        
        * The `bounds` variable is parametrization-dependent!
        """
        ops = DEFAULT_QNET_OPS
        ops.update(algo_ops)
        theta_0 = self.PQCs[0].gen_rand_data(1, include_metric=False).squeeze()
        n_param = theta_0.shape[0]
        
        if self.mode == Q_MODE_NM:
            bounds = [(0, 2*pi)] * n_param
            result = minimize(self.param_to_quantum_loss, 
                            theta_0, bounds=bounds, method='Nelder-Mead', 
                            options={'disp': ops['disp'], 
                                    'return_all': False, 
                                    'initial_simplex': None, 
                                    'xatol': 0.0001, 
                                    'fatol': 0.0001, 
                                    'adaptive': ops['adaptive']})
            
            if print_log:
                print(f"Optimization {'SUCCEEDED' if result.success else 'FAILED'}: exit code {result.status}")
                print(f"Message from solver: {result.message}")
                print(f"Final {'non-' if not self.regularize else ''}regularized loss value: {result.fun}")
            return result.x

        elif self.mode == Q_MODE_GD:
            h = ops['h'] # finite difference parameter
            theta = theta_0
            num_epochs = int(ops['num_epochs'])
            max_iter = int(ops['max_iter'])
            abs_grads = np.zeros(num_epochs)
            completion_times = np.zeros(num_epochs)
            idxs_chosen = np.zeros(num_epochs)
            
            for i in range(num_epochs):
                lr = eta # an adjustable learning rate
                idx = npr.randint(n_param) # indexes the parameter we will do coordinate FDGD on in this epoch
                num_steps = 0
                idxs_chosen[i] = idx
                
                while num_steps < max_iter:
                    theta_perturbed = theta.clone()
                    theta_perturbed[idx] += theta_perturbed[idx] + h
                    
                    # Evaluate the loss at the (un-)perturbed state and take a finite difference
                    fd_stochastic_deriv = (self.param_to_quantum_loss(theta_perturbed) - self.param_to_quantum_loss(theta)) / h
                    theta[idx] -= lr * fd_stochastic_deriv # take a step
                    num_steps += 1
                    if abs(fd_stochastic_deriv) <= atol:
                        break
                    if num_steps == max_iter:
                        print(f"[Epoch {i+1}] FDSGD failed to converge on parameter {idx} within {max_iter} steps. Grad = {fd_stochastic_deriv}")
                    
                    # Exponentially adjust the learning rate if we're in roughly in the order of magnitude
                    if abs(fd_stochastic_deriv) / atol <= 10:
                        lr = max(MINIMUM_LR, 0.95 * lr)
                        
                # Compute some statistics
                abs_grads[i] = abs(fd_stochastic_deriv)
                completion_times[i] = num_steps
                if print_log:
                    print("Run Statistics:")
                    print(f"Avg steps per epoch = {np.mean(completion_times)} with deviation {np.std(completion_times)}")
                    print(f"Avg abs grad = {np.mean(abs_grads)} with deviation {np.std(abs_grads)}")
                
            return theta.numpy(), idxs_chosen
        
        
    
    