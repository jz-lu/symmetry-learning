from ___constants import CNET_CONV_NCHAN, CNET_HIDDEN_DIM, PARAM_PER_QUBIT_PER_DEPTH
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

* Open problem: at some point we will obviously need to scale CNET_HIDDEN_DIM
* as a function of the number of qubits. What's the scaling function?

* Open problem: do we need to add another convolutional layer to treat the depth?
"""

class CNet(nn.Module):
    def __init__(self, num_qubits, depth=0):
        super(CNet, self).__init__() # initialize torch nn
        self.depth = depth
        self.conv1 = nn.Conv3d(1, CNET_CONV_NCHAN, (1,1,PARAM_PER_QUBIT_PER_DEPTH))
        self.conv2 = nn.Conv3d(CNET_CONV_NCHAN, CNET_CONV_NCHAN, 1)
        self.linear1 = nn.Linear(CNET_CONV_NCHAN * num_qubits * (depth+1), CNET_HIDDEN_DIM)
        self.linear2 = nn.Linear(CNET_HIDDEN_DIM, CNET_HIDDEN_DIM)
        self.linear3 = nn.Linear(CNET_HIDDEN_DIM, 1)
        self.model = self # hack
        self.batch_size = 100
        self.loss_func = nn.MSELoss()
        self.num_qubits = num_qubits
        self.train_q = []
        print(f"Classical deep net of circuit depth {self.depth} initialized.")
    
    def forward(self, param):
        """
        * Parametrization-dependent function.
        """
        x = param.view(-1, self.num_qubits, self.depth+1, PARAM_PER_QUBIT_PER_DEPTH)
        x = t.unsqueeze(x, 1) # to give CNN the trivial 1-input channel
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, CNET_CONV_NCHAN * self.num_qubits * (self.depth+1))
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return x

    def train_SGD(self, datum, eta=1e-2):
        """
        Train the network with a single datum rather than a batch over epochs.
        Confusingly, SGD here stands for singleton gradient descent; nothing is
        stochastic in this process, as `datum` is provided to us by some algorithm.
        """
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=eta)
        true_metric = datum[-1]
        self.optimizer.zero_grad()
        estimated_metric = self.model(datum[:-1]).squeeze()
        loss = self.loss_func(estimated_metric, true_metric)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, train_data, nepoch=2000, eta=1e-2, loss_window=10, print_log=False):
        """
        Train the net with learning rate `eta` for `nepochs` epochs.
        """
        true_metric = train_data[:,-1] # True metric value
        sz = train_data.size(0)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=eta)
        self.losses = np.zeros(nepoch // loss_window)
        local_losses = [0] * loss_window
        if print_log:
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
            if epoch % 1000 == 0 and print_log:
                print('.', end='', flush=True)
        if print_log:
            print("") # newline
        return self.losses
    
    def test(self, test_data):
        """Apply the CNet to a test dataset. Returns the cross-validation MSE"""
        estimated_metric = self.model(test_data[:,:-1]).squeeze()
        true_metric = test_data[:,-1]
        return self.loss_func(estimated_metric, true_metric)

    def run(self, data):
        """Run the CNet on a dataset and return the estimated metric"""
        return self.model(data)
    
    def run_then_train(self, data, nepoch=2000, eta=1e-2, loss_window=10):
        """
        First evaluate the batch of data using the network, then train it. 
        Used for QNet-CNet interaction in the HQNet training scheme.
        Returns: (estimated metric, training loss).
        """
        assert len(data.shape == 2)
        return self.run(data[:,:-1]), self.train(data, nepoch=nepoch, eta=eta, loss_window=loss_window)
    
    def run_then_train_SGD(self, datum):
        """
        First evaluate the data using the network on a single datum, then
        train it using `train_SGD`.
        """
        assert len(datum.shape) == 1
        return self.run(datum[:-1]), self.train_SGD(datum)

    def run_then_enq(self, datum):
        """
        Run the network on the datum, then enqueue onto a queue
        for training. The queue is trained as a batch when desired.
        """
        assert len(datum.shape) == 1
        self.train_q.append(datum)
        return self.run(datum[:-1])
        
    def flush_q(self, nepoch=2000, eta=1e-2, loss_window=10, print_log=False):
        """
        Flush the training queue by training it all as a batch.
        """
        losses = self.train(np.array(self.train_q), 
                          nepoch=nepoch, eta=eta, 
                          loss_window=loss_window, 
                          print_log=print_log)
        self.train_q = [] # clear the queue
        return losses
        
