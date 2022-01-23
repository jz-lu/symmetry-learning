import numpy as np
from scipy.optimize import minimize as nelder_mead
from math import pi
from __loss_funcs import KL
from __class_CNet import CNet
from __class_PQC import PQC


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
        print("Hybrid quantum net initialized -- Hello world!")
    
    def __quantum_loss_metric(self, classical_loss_tensor):
        """
        The classical loss tensor is of order 3: (num bases, batch size, 2)
        where the last index are pairs (true metric, estimated metric). The 
        quantum loss metric takes a positive monotonic map over the estimated metric,
        then takes its difference with the true metric. It then sums over the bases.
        """
        return np.sum(np.apply_along_axis(self.qloss, 2, classical_loss_tensor), 0)
    
    def param_to_quantum_loss(self, p):
        """
        Function mapping a parameter to the quantum loss.
        `p` is a parametrization of the PQC.
        
        Note: to train the CNet we will need a concatenation of the parameter and the 
        true metric. See `quantum_loss_metric()` for documentation on `classical_loss_tensor`.
        """
        classical_loss_tensor = np.zeros((self.num_bases, 2))
        classical_loss_tensor[:,0] = np.array([qc.evaluate_true_metric(p) for qc in self.PQCs])
        classical_loss_tensor[:,1] = np.array([cnet.run_then_train_SGD(
                                                    np.concatenate((p, classical_loss_tensor[i,0])))[0]
                                                for i, cnet in enumerate(self.CNets)]) # estimated metric
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
        ? If so, how can we resolve this? Note that a pre-training phase interefers
        ? with our proof of regularizability. Perhaps we can improve the CNet by using
        ? a RNN or LSTM layer?
        
        * The `bounds` variable is parametrization-dependent!
        """
        theta_0 = np.array(
                        [
                            qc.gen_rand_data(1, include_metric=False).squeeze() 
                        for qc in self.PQCs
                        ]
                       ) # shape: (num bases,)
        bounds = [(0, 2*pi)] * (self.L * 3)
        result = nelder_mead(self.param_to_quantum_loss, 
                                   theta_0, disp=disp, 
                                   adaptive=adaptive, 
                                   bounds=bounds)
        
        print(f"Optimization {'SUCCEEDED' if result.success else 'FAILED'}: exit code {result.status}")
        print(f"Message from solver: {result.message}")
        print(f"Final regularized loss value: {result.fun}")
        return result.x
        
    
    