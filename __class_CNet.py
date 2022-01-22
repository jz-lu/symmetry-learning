import torch as t
import torch.nn as nn
from torch.nn import functional as F
import random
import numpy as np

"""
The CNet component of the HQ Net. 
In iteration 1: CNet is a stochastic function approximator
mapping distributions to the approximate metric.

Implementation followed from: https://pyt.org/tutorials/beginner/blitz/neural_networks_tutorial.html.

Currently, the metric we use is the: KL Divergence.
"""
CNET_HIDDEN_DIM = 100
CNET_CONV_NCHAN = 4
def my_loss(x):
    return x * t.pow(t.sin(x), 2)

class CNet(nn.Module):
    def __init__(self, num_qubits):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(1, CNET_CONV_NCHAN, (1,3))
        self.conv2 = nn.Conv2d(CNET_CONV_NCHAN, CNET_CONV_NCHAN, (1,1))
        self.linear1 = nn.Linear(12, CNET_HIDDEN_DIM)
        self.linear2 = nn.Linear(CNET_HIDDEN_DIM, CNET_HIDDEN_DIM)
        self.linear3 = nn.Linear(CNET_HIDDEN_DIM, 1)
        self.model = self # hack
        self.batch_size = 100
        self.loss_func = nn.MSELoss()
    
    def forward(self, param):
        x = param.view(-1, 3, 3)
        x = t.unsqueeze(x, 1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, 12)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return x
    
    def train(self, train_data, nepoch=2000, eta=1e-2, loss_window=10):
        """Train the net with learning rate `eta` for `nepochs` epochs"""
        true_metric = train_data[:,-1] # True metric value
        sz = train_data.size(0)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=eta)
        self.losses = np.zeros(nepoch // loss_window)
        local_losses = [0] * loss_window
        print("Training", flush=True, end='')
        
        for epoch in range(nepoch):
            self.optimizer.zero_grad()
            batch_idxs = random.sample(range(sz), self.batch_size)
            batch = train_data[batch_idxs].clone()
            estimated_metric = self.model(batch[:,:-1]).squeeze()
            loss = self.loss_func(estimated_metric, true_metric[batch_idxs])
            if epoch % loss_window == 0 and epoch > 0:
                self.losses[epoch // loss_window] = np.mean(local_losses)
                local_losses = [0] * loss_window
            elif epoch == 0:
                self.losses[0] = loss.item()
            else:
                local_losses[epoch % loss_window] = loss.item()
            loss.backward()
            self.optimizer.step()
            if epoch % 1000 == 0:
                print('.', end='', flush=True)
        print("") # newline
        return self.losses
    
    def test(self, test_data):
        """Apply the CNet to a test dataset. Returns the cross-validation MSE"""
        estimated_metric = self.model(test_data[:,:-1]).squeeze()
        true_metric = test_data[:,-1]
        return self.loss_func(estimated_metric, true_metric)
