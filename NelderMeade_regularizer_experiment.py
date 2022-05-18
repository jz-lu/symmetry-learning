from __helpers import qubit_expansion, rand_basis
from ___constants import PARAM_PER_QUBIT_PER_DEPTH
from __loss_funcs import KL
from __class_BasisTransformer import BasisTransformer
from __class_PQC import PQC
from __class_HQNet import HQNet
import numpy as np
from math import pi
from qiskit.quantum_info import Statevector
#%matplotlib inline

# Useful additional packages 
import numpy as np
from math import pi

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import torch







#lets create GHZ state
circ = QuantumCircuit(3)
# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(0)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(0, 1)
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(0, 2)
# Set the intial state of the simulator to the ground state using from_int
state = Statevector.from_int(0, 2**3)
state1 = state.copy()
# Evolve the state by the quantum circuit
state = state.evolve(circ)
print(state)
print(state1)


CIRCUIT_DEPTH = 0 # Depth of the parameterized quantum circuit
STATE_TYPE = 'GHZ'
NUM_QUBITS = 3
USE_REGULARIZER = False


"""[Notes on the states]

Confusingly, the cluster state is actually NUM_QUBITS^2 qubits rather than NUM_QUBITS. 
This is due to the way the circuit is designed. The rest are as you would expect.
"""

if STATE_TYPE == 'GHZ':
    # Prepare: GHZ State (from: Q-circuit)
    from GHZ_generator import GHZ_state_circuit
    state = Statevector.from_int(0, 2**NUM_QUBITS)
    qc = GHZ_state_circuit(L=NUM_QUBITS)
    print(qc)
    state = state.evolve(qc)
elif STATE_TYPE == 'XY':
    # Prepare: XY(L) (from: ED)
    from XY_generator import xy_ground_state
    state = Statevector(xy_ground_state(NUM_QUBITS).numpy())
elif STATE_TYPE == 'Cluster':
    # Prepare cluster(L) (from: Q-circuit)
    from cluster_generator import cluster_state_circuit
    state = Statevector.from_int(0, 2**(NUM_QUBITS**2))
    qc = cluster_state_circuit(NUM_QUBITS)
    print(qc)
    state = state.evolve(qc)
else:
    raise TypeError("Invalid state type specified.")
param_shape = (state.num_qubits, CIRCUIT_DEPTH+1, PARAM_PER_QUBIT_PER_DEPTH)

# Visualize the distribution
plt.bar(qubit_expansion(state.num_qubits), state.probabilities())
plt.title(f"{NUM_QUBITS}-{STATE_TYPE} State Distribution (z-basis)")
plt.show()



#function that apply the KS test to two probability list
def KS(P1, P2):
    assert len(P1) == len(P2)
    cdf1 = [P1[0]]
    cdf2 = [P2[0]]
    for i in range(len(P1)-1):
        cdf1.append(cdf1[i] + P1[i+1])
        cdf2.append(cdf2[i] + P2[i+1])
    difference = torch.tensor(cdf1) - torch.tensor(cdf2)

    #print(difference)
    return torch.pow(difference, 2).sum()#difference.abs().max().item()

print(KS([0.2, 0.8], [0.5, 0.5]))


def KL(P1, P2):
    Q = torch.tensor(P1)
    Q = Q + 0.00001 * torch.ones(Q.size())
    P = torch.tensor(P2)
    P = P + 0.000001 * torch.ones(Q.size())
    #print(Q)
    #tens = torch.div(P, Q)
    #tens = 
    return (torch.log(torch.div(P, Q)) * P).sum().item()

print(KL([0, 1], [0.5, 0.5]))


from torch.nn import functional as F
import torch.nn as nn

#This is our KS network duh
class KS_net(nn.Module):
    def __init__(self):
        super(KS_net, self).__init__()
        self.linear1 = nn.Linear(9, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 1)
    
    def forward(self, param):
        x = F.leaky_relu(self.linear1(param))
        #print(x)
        x = F.leaky_relu(self.linear2(x))
        #print(x)
        x = F.leaky_relu(self.linear3(x))
        return x


class CONV3(nn.Module):
    def __init__(self):
        super(CONV3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (1,3))
        self.conv2 = nn.Conv2d(4, 20, (1,1))
        self.linear1 = nn.Linear(60, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20,1)
    
    def forward(self, param):
        #print(param.size())
        x = param.view(-1, 3,3)
        
        x = torch.unsqueeze(x, 1)
       #print(x.size())
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, 60)
        x = F.leaky_relu(self.linear1(x))
        #print(x)
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return x




from qiskit.algorithms.optimizers import (
    ADAM, AQGD, CG, GradientDescent,
    NELDER_MEAD, NFT, SPSA, TNC, 
    ESCH, ISRES, DIRECT_L
)



#this is where we try to learn the symmetry
import random
import scipy


ls1 = [] #loss with regularization
ls2 = [] #nn training loss

class SymFinder():
  def __init__(self, maxiter1,  lr2, gradient_step2):
    self.parameters = 0.5 * torch.ones(3,3)
    #parameters[i,0] is the theta for ith qubit, 1 is \phi, 2 is lambda
    self.original_state = state
    self.transformed_state = None
    self.losses = []
    self.known_symmetries = []
     
    self.lr2 = lr2 #this is the lr for exploring
    self.gradient_step2 = gradient_step2

    #now these are the nn attributes
    self.data = torch.zeros(30000, 10) #data for training KS_net, 9 + 1 = param + value
    self.data.requires_grad = False
    self.learn_rate = 0.01 #learning rate for training KS_net
    self.model = CONV3()
    self.model = self.model.float()
    self.batch_size = 50 #batch for SGD
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learn_rate)
    self.loss_func = nn.MSELoss()
    self.memory_pointer = 0 #tell where to store new data
    self.valid_range = 0#how much valid data is in the tensor, dont want to learn 00000

    self.sym_optimizer = NELDER_MEAD(maxiter=maxiter1, maxfev=maxiter1, adaptive=True, disp=False)
    self.KS_threshold = 0.001
    #self.reg_optimizer = NELDER_MEAD(maxiter=maxiter2, maxfev=maxiter2, adaptive=True, disp=True)#, xatol = 0, tol = -100)

  
  #return a transformed state according to parameter
  def transform(self, p):
    q = QuantumRegister(3)
    qc = QuantumCircuit(q)
    qc.u(p[0,0].item(),p[0,1].item(),p[0,2].item(),q[0])
    qc.u(p[1,0].item(),p[1,1].item(),p[1,2].item(),q[1])
    qc.u(p[2,0].item(),p[2,1].item(),p[2,2].item(),q[2])
    return self.original_state.copy().evolve(qc)
  

  
  def change_basis(self, state1, state2):
    q = QuantumRegister(3)
    qc = QuantumCircuit(q)
    qc.u(0*pi/10, 0, 0, q[0])
    qc.u(1*pi/10, 0, 0, q[1])
    qc.u(0*pi/10, 0, 0, q[2])
    return state1.copy().evolve(qc), state2.copy().evolve(qc)

  #return the loss from KS test of original vs another state
  def calculate_loss(self, state2, param):
    #get the probability in the original basis
    P1 = self.original_state.probabilities()
    P2 = state2.probabilities()
    #now we calculate probability in another basis
    new_state1, new_state2 = self.change_basis(self.original_state, state2)
    Q1 = new_state1.probabilities()
    Q2 = new_state2.probabilities()
    #lets add regularizer from our nn
    true_loss = KL(P1, P2) +  KL(Q1, Q2)
    v = self.model(param.view(9).float()).detach_()
    #addon = 20 * torch.tanh(1/((v-true_loss)**2))
    addon = float(-(v-true_loss)**2)
    #v = 0
    return [addon, true_loss]
  

  def calculate_loss1(self, state2, param):
    #get the probability in the original basis
    P1 = self.original_state.probabilities()
    P2 = state2.probabilities()
    #now we calculate probability in another basis
    new_state1, new_state2 = self.change_basis(self.original_state, state2)
    Q1 = new_state1.probabilities()
    Q2 = new_state2.probabilities()
    #lets add regularizer from our nn
    true_loss = KL(P1, P2) +  KL(Q1, Q2)
    return [true_loss, true_loss]





  def true_loss(self, param_vec):
    param = torch.tensor(param_vec).view(3,3)
    state2 = self.transform(param)
    cur_true_loss = self.calculate_loss1(state2, param)[1]
    self.data[self.memory_pointer, 0:9] = torch.tensor(param_vec)
    self.data[self.memory_pointer, 9] = cur_true_loss
    self.memory_pointer = (self.memory_pointer + 1) % self.data.size(0)
    self.valid_range = self.valid_range + 1
    return cur_true_loss
  

  def regularizer_loss(self, param_vec):
    param = torch.tensor(param_vec).view(3,3)
    state2 = self.transform(param)
    reg_loss = self.calculate_loss(state2, param)[0]
    #return np.pi/2 + np.arctan(reg_loss)
    return reg_loss

    
  

  def update_param_regularize(self):
    #calculate the gradient using good old finite difference:
    cur_state = self.transform(self.parameters)
    cur_loss = self.calculate_loss(cur_state, self.parameters)[0]
    cur_true_loss = self.calculate_loss(cur_state, self.parameters)[1]
    #lets store this data
    self.data[self.memory_pointer, 0:9] = self.parameters.reshape(9)
    self.data[self.memory_pointer, 9] = cur_true_loss
    self.memory_pointer = (self.memory_pointer + 1) % self.data.size(0)
    self.valid_range = self.valid_range + 1
    grad = torch.zeros(3,3)
    for i in range(3):
        for j in range(3):
            new_param = self.parameters.clone()
            new_param[i,j] = new_param[i,j] + self.gradient_step2
            new_param = torch.fmod(new_param, 1000*pi)
            new_state = self.transform(new_param)
            new_loss = self.calculate_loss(new_state, new_param)[0]
            grad[i,j] = (new_loss - cur_loss) / self.gradient_step2
    #update the parameters:
    self.parameters = torch.fmod(self.parameters - self.lr2 * grad, 2*pi)
    
    #self.parameters.requires_grad = False
    #self.parameters[:, 2] = torch.zeros(3)
    ls1.append(cur_loss)
    return cur_true_loss





  def update_KS_net(self):
    #this function use the data gathered to update KS_net parameters
    for i in range(100):
      self.optimizer.zero_grad()
      num = min(self.valid_range, self.data.size(0))
      #print(num, self.batch_size)
      indices = random.sample(range(num), min(self.batch_size, num))
      sampled_data = self.data[indices].clone()
      target = sampled_data[:, 9] #actual KS_value
      current = self.model(sampled_data[:, 0:9]).squeeze()
      loss = self.loss_func(current, target)
      ls2.append(loss.item())
      #print(loss)
      loss.backward()
      self.optimizer.step()
    

  


    
  



  def symfinding(self):
    #self.parameters = torch.randint(0, 100, (3, 3)) * 2 * pi /100
    bounds = [(0, 2*pi)] * 9
    point, value, nfev = self.sym_optimizer.optimize(9, 
                                                  self.true_loss, 
                                                  initial_point = self.parameters.view(9), 
                                                  variable_bounds = bounds)
    self.parameters = torch.tensor(point).view(3,3)
    if value < self.KS_threshold:
      self.known_symmetries.append(torch.tensor(point).view(3,3))   
    #print(self.true_loss(self.parameters.view(9)))
    #print(self.parameters)
    #print("_________________")     

  def exploring(self):
  #self.parameters = torch.randint(0, 100, (3, 3)) * 2 * pi /100
    for i in range(60):
      cur_true_loss = self.update_param_regularize()
      self.losses.append(cur_true_loss)
      #if cur_true_loss < self.KS_threshold:
        #self.known_symmetries.append(self.parameters.clone())
        #break



  




  def train(self):
    for i in range(130):
      #print(self.parameters)
      
      self.symfinding()
      #self.parameters = torch.randint(0, 100, (3, 3)) * 2 * pi /100
      self.exploring()
      self.update_KS_net()
      print(i)
      #self.model2.load_state_dict(self.model.state_dict())


  def current_matrix(self):
    p = self.parameters
    #p[0,:] = torch.tensor([3.14159, 0,3.1415926])
    u0 = torch.tensor([[torch.cos(p[0,0]/2), -torch.exp(-p[0,2]*1j)*torch.sin(p[0,0]/2)],\
                    [torch.exp(p[0,1]*1j)*torch.sin(p[0,0]/2), torch.exp((p[0,1] + p[0,2])*1j)*torch.cos(p[0,0]/2)]])
    u1 = torch.tensor([[torch.cos(p[1,0]/2), -torch.exp(-p[1,2]*1j)*torch.sin(p[1,0]/2)],\
                    [torch.exp(p[1,1]*1j)*torch.sin(p[1,0]/2), torch.exp((p[1,1] + p[1,2])*1j)*torch.cos(p[1,0]/2)]])
    u2 = torch.tensor([[torch.cos(p[2,0]/2), -torch.exp(-p[2,2]*1j)*torch.sin(p[2,0]/2)],\
                    [torch.exp(p[2,1]*1j)*torch.sin(p[2,0]/2), torch.exp((p[2,1] + p[2,2])*1j)*torch.cos(p[2,0]/2)]])
    return [u0, u1, u2]
        
        
        
        

import numpy as np

finder = SymFinder(5000, 1,0.01)





def matrix(p):
    #p[0,:] = torch.tensor([3.14159, 0,3.1415926])
    u0 = torch.tensor([[torch.cos(p[0,0]/2), -torch.exp(-p[0,2]*1j)*torch.sin(p[0,0]/2)],\
                    [torch.exp(p[0,1]*1j)*torch.sin(p[0,0]/2), torch.exp((p[0,1] + p[0,2])*1j)*torch.cos(p[0,0]/2)]])
    u1 = torch.tensor([[torch.cos(p[1,0]/2), -torch.exp(-p[1,2]*1j)*torch.sin(p[1,0]/2)],\
                    [torch.exp(p[1,1]*1j)*torch.sin(p[1,0]/2), torch.exp((p[1,1] + p[1,2])*1j)*torch.cos(p[1,0]/2)]])
    u2 = torch.tensor([[torch.cos(p[2,0]/2), -torch.exp(-p[2,2]*1j)*torch.sin(p[2,0]/2)],\
                    [torch.exp(p[2,1]*1j)*torch.sin(p[2,0]/2), torch.exp((p[2,1] + p[2,2])*1j)*torch.cos(p[2,0]/2)]])
    return [u0, u1, u2]



  

data = torch.ones(50, 50, 100, 3, 3)






for i in range(50):
    for j in range(50):
        lr = (i+1) * (1/50)
        gs = (j+1) * (0.1/50)
        finder = SymFinder(5000, lr2 = lr, gradient_step2 = gs)
        finder.train()
        for k in range(len(finder.known_symmetries)):
            data[i,j,k % 100,: ,: ] = finder.known_symmetries[k]
        print(i,j)
        torch.save(data, 'NM_regularizer_data.pt')

torch.save(data, 'NM_regularizer_data.pt')
  

  
