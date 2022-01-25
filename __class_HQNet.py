from __loss_funcs import KL
from __class_CNet import CNet
from __class_PQC import PQC
import numpy as np
from scipy.optimize import minimize
import torch.nn as nn
import torch as t
from math import pi


"""
HQNet: Hybrid quantum network. This is the full scheme for learning the symmetries
of a given quantum state. It combines a parametrized quantum circuit (PQC)
and a classical deep network (CNet), which learns to estimate the loss within 
the HQNet to regularize against finding known symmetries.

Currently built to perform either gradient descent (mode=gd) or Nelder-Mead
simplex search (mode=nm).
"""

class HQNet:
    def __init__(self, state, bases, metric_func=KL, mode='nm', regularize=False):
        """
        Use a quantumfication of loss metric `metric_func` 
        over each basis in the list of bases `bases`.
        """
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
        self.mode = mode
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
        `p_vec` is a vectorized parametrization (size: dim theta * num bases) of the PQC.
        
        Note: to train the CNet we will need a concatenation of the parameter and the 
        true metric. See `quantum_loss_metric()` for documentation on `classical_loss_tensor`.
        """
        p_vec = t.from_numpy(p_vec)
        p_tens = t.zeros((self.num_bases, p_vec.size()[0] + 1))
        for i in range(self.num_bases):
            p_tens[i,:-1] = p_vec.clone()
        classical_loss_tensor = t.zeros((self.num_bases, 2))
        classical_loss_tensor[:,0] = t.tensor([qc.evaluate_true_metric(p_vec) for qc in self.PQCs])
        classical_loss_tensor[:,1] = t.tensor([cnet.run_then_train_SGD(p)[0]
                                                for p, cnet in zip(p_tens, self.CNets)]) # estimated metric
        return self.__quantum_loss_metric(classical_loss_tensor)
    
    def find_potential_symmetry(self, disp=False, adaptive=False):
        """
        Run the optimization algorithm to obtain the maximally symmetric
        parameter, regularizing with the CNet. Train the CNet in parallel.
        Start at a random initial parameter `theta_0`.
                
        For singleton datum training only. We have not yet come up with a batch
        training scheme that does not use the (highly inefficient) finite
        difference method. The current method is: Nelder-Mead simplex search.
        
        Prints convergence messages if `disp` is True. Uses an adaptive version
        of Nelder-Mead if `adaptive` is True, which has been shown to be successful
        for high-dimensional simplex search (our case!).
        
        ? Open question: will the CNet performance get worse when the batch is 
        ? a local path in the space, not a randomly sampled set of parameters?
        ? (We want it to still be able to interpolate very well, but we DON'T 
        ? want it to extrapolate well. That's the point of regularizing.)
        
        * The `bounds` variable is parametrization-dependent!
        """
        theta_0 = self.PQCs[0].gen_rand_data(1, include_metric=False).squeeze()
        bounds = [(0, 2*pi)] * (self.L * 3)
        result = minimize(self.param_to_quantum_loss, 
                          theta_0, bounds=bounds, method='Nelder-Mead', 
                          options={'disp': disp, 
                                   'return_all': False, 
                                   'initial_simplex': None, 
                                   'xatol': 0.0001, 
                                   'fatol': 0.0001, 
                                   'adaptive': adaptive})
        
        print(f"Optimization {'SUCCEEDED' if result.success else 'FAILED'}: exit code {result.status}")
        print(f"Message from solver: {result.message}")
        print(f"Final {'non-' if not self.regularize else ''}regularized loss value: {result.fun}")
        return result.x
        
    
    