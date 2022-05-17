import torch as t
import numpy as np
from torch.linalg import eig

import numpy.linalg as LA

'''Device where we are running'''
#defining the device
if t.cuda.is_available():
    device = t.device("cuda:0") 
    print("Running on the GPU")
else:
    device = t.device("cpu")
    print("Running on the CPU")


#Matrices we need
Sx = t.tensor([[0,1.],[1,0]], dtype=t.cfloat).to(device)
Sy = t.tensor([[0,-1j],[1j,0]], dtype=t.cfloat).to(device)
Sz = t.tensor([[1,0],[0,-1]], dtype=t.cfloat).to(device)
Sp = t.tensor([[0,1],[0,0]], dtype=t.cfloat).to(device)
Sm = t.tensor([[0,0],[1,0]], dtype=t.cfloat).to(device)
nop = t.tensor([[0,0],[0,1]], dtype=t.cfloat).to(device)
Id2 = t.tensor([[1,0],[0,1]], dtype=t.cfloat).to(device)


def add_op(ops, Js, sites, L): 
    l = [Id2 for i in range(L)]
    for i in range(len(sites)): 
        l[sites[i]] = Js[i]*ops[i]
    m = l[0]
    for i in range(1,L): m = t.kron(m,l[i])
    return m




Omega = 2*np.pi*2 #MHz
Delta = 2*np.pi*20 #MHz

C = 2*np.pi*50


#For details see doi:10.1038/nature24622

class Ryrberg1D(): 
    def __init__(self, N):
        self.N = N
        self.get_H()
        
    '''Do not call outside constructor'''
    def get_H(self): 
        self.H = t.zeros((2**self.N, 2**self.N), dtype=t.cfloat).to(device)
        for i in range(self.N): 
            self.H += add_op([nop],[-Delta], [i], self.N)
            self.H += add_op([Sx],[Omega/2], [i], self.N)
            for j in range(i+1, self.N): 
                self.H += add_op([nop, nop], [C/(abs(j-i)**6) , 1], [i,j], self.N)
        return
    
    '''Time evolution with Hamiltonian'''
    def evolution(self, t): 
        return t.matrix_exp(self.H*t*1j)
    
    '''Diagonalize Hamiltonian and return k lowest energy states'''
    def get_ground_state(self, k=1): 
        res = LA.eigh(self.H)
        return res[0][0:k], res[1][:,0:k]

def rydberg_ground_state(N, V=2*np.pi*414.3):
    """Generate XY gs on L qubits. See Hamiltonians.ipynb for parameter docs"""
    gse, gs = Ryrberg1D(N).get_ground_state()
    print(f"Ground state energy of Rydberg({N}): {gse.item()}")
    return gs