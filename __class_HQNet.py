import torch as t
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from __loss_funcs import KL
from __class_CNet import CNet
from __class_PQC import PQC
from scipy.optimize import minimize as nelder_mead

"""
HQNet: Hybrid quantum network. This is the full scheme for learning the symmetries
of a given quantum state. It combines a parametrized quantum circuit (PQC)
and a classical deep network (CNet), which learns to estimate the loss within 
the HQNet to regularize against finding known symmetries.

Current parameter-space optimization algorithm: Nelder-Mead simplex search.
"""
HQN_BATCH_SIZE = 50
class HQNet:
    def __init__(self, state, bases, metric_func=KL):
        """
        Use a quantumfication of loss metric `metric_func` 
        over each basis in the list of bases `bases`.
        """
        self.CNets = [CNet(
                            state.num_qubits
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
        print("Hybrid quantum net initialized -- Hello world!")
    
    def quantum_loss_metric(self, classical_loss_tensor):
        """
        The classical loss tensor is of order 3: (batch_size, num bases, 2)
        where the last index are pairs (true metric, estimated metric). The 
        quantum loss metric takes a positive monotonic map over the estimated metric,
        then takes its difference with the true metric. It then sums over the bases.
        """
        classical_loss_tensor = classical_loss_tensor.numpy()
        return np.sum(np.apply_along_axis(self.qloss, 2, classical_loss_tensor), 1)
    
    def find_potential_symmetry(self, q_nepoch=200, q_eta=1e-2,
                                c_nepoch=3000, c_eta=1e-2, 
                                batch_size=HQN_BATCH_SIZE):
        """
        Run the optimization algorithm to obtain the maximally symmetric
        parameter, regularizing with the CNet. Train the CNet in parallel.
        
        * Open question: will the CNet performance get worse when the batch is 
        * a local path in the space, not a randomly sampled set of parameters?
        """
        params = PQC.gen_rand_data(batch_size)
        
        for i in range(q_nepoch):
            estimated_metric = CNet.run_then_train(params, nepoch=c_nepoch, eta=c_eta)
            # TODO do for each basis then sum!!
    
    