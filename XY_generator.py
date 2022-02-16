# Adapted from Rodrigo's XY Hamiltonian ED GS generator

import torch as t
import numpy as np
from torch.linalg import eig

'''Device where we are running'''
#defining the device
if t.cuda.is_available():
    device = t.device("cuda:0") 
    print("Running on the GPU")
else:
    device = t.device("cpu")
    print("Running on the CPU")
    
# Matrices we need
Sx = t.tensor([[0,1.],[1,0]], dtype=t.cfloat).to(device)
Sy = t.tensor([[0,-1j],[1j,0]], dtype=t.cfloat).to(device)
Sz = t.tensor([[1,0],[0,-1]], dtype=t.cfloat).to(device)
Sp = t.tensor([[0,1],[0,0]], dtype=t.cfloat).to(device)
Sm = t.tensor([[0,0],[1,0]], dtype=t.cfloat).to(device)
Id2 = t.tensor([[1,0],[0,1]], dtype=t.cfloat).to(device)

# Function to add terms to Hamiltonians
def add_op(ops, Js, sites, L): 
    l = [Id2 for i in range(L)]
    for i in range(len(sites)): 
        l[sites[i]] = Js[i]*ops[i]
    m = l[0]
    for i in range(1,L): m = t.kron(m,l[i])
    return m

class XY: 
    def __init__(self, J, n, h, L, bc = 'periodic', hx=1e-6): 
        self.J = J # Interaction strength
        self.n = n # eta = anisotropy
        self.h = h # transverse field strength
        self.L = L # number of qubits
        self.bc = bc # boundary conditions
        self.hx = hx # avoid hx = 0
        self.Jx = J*(1+n)/2
        self.Jy = J*(1-n)/2
        
        self.get_H()
    
    '''Do not call outside constructor'''
    def get_H(self): 
        self.H = t.zeros((2**self.L, 2**self.L), dtype=t.cfloat).to(device)
        for i in range(self.L): 
            self.H += add_op([Sz],[self.h], [i], self.L)
            self.H += add_op([Sx],[self.hx], [i], self.L)
            if self.bc == 'periodic' or i<self.L-1:
                self.H -= add_op([Sx, Sx], [self.Jx, 1], [i,(i+1)//self.L], self.L)
                self.H -= add_op([Sy, Sy], [self.Jy, 1], [i,(i+1)//self.L], self.L)
        return
    
    '''Time evolution with Hamiltonian'''
    def evolution(self, t): 
        return t.matrix_exp(self.H*t*1j)
    
    '''Diagonalize Hamiltonian and return k lowest energy states'''
    def get_ground_state(self, k=1): 
        res = eig(self.H)
        return res[0][0:k], res[1][:,0:k]

def xy_ground_state(L, J=1, n=1, h=5e-1):
    """Generate XY gs on L qubits. See Hamiltonians.ipynb for parameter docs"""
    gse, gs = XY(J, n, h, L).get_ground_state()
    print(f"Ground state energy of XY({L}): {gse.item()}")
    return gs