import torch
import numpy as np
import matplotlib.pyplot as plt

#data = torch.load('regularizer_data.pt')

data1 = torch.load('NM_regularizer_data1.pt')
data0 = torch.load('NM_regularizer_data0.pt')
data2 = torch.load('NM_regularizer_data2.pt')
data3 = torch.load('NM_regularizer_data3.pt')


def matrix(p):
    #p[0,:] = torch.tensor([3.14159, 0,3.1415926])
    u0 = torch.tensor([[torch.cos(p[0,0]/2), -torch.exp(-p[0,2]*1j)*torch.sin(p[0,0]/2)],\
                    [torch.exp(p[0,1]*1j)*torch.sin(p[0,0]/2), torch.exp((p[0,1] + p[0,2])*1j)*torch.cos(p[0,0]/2)]])
    u1 = torch.tensor([[torch.cos(p[1,0]/2), -torch.exp(-p[1,2]*1j)*torch.sin(p[1,0]/2)],\
                    [torch.exp(p[1,1]*1j)*torch.sin(p[1,0]/2), torch.exp((p[1,1] + p[1,2])*1j)*torch.cos(p[1,0]/2)]])
    u2 = torch.tensor([[torch.cos(p[2,0]/2), -torch.exp(-p[2,2]*1j)*torch.sin(p[2,0]/2)],\
                    [torch.exp(p[2,1]*1j)*torch.sin(p[2,0]/2), torch.exp((p[2,1] + p[2,2])*1j)*torch.cos(p[2,0]/2)]])
    return [u0, u1, u2]



def diag_or_not(p):
    diag = torch.abs(matrix(p)[0][0,0]).item()
    off_diag = torch.abs(matrix(p)[0][0,1]).item()
    if diag <= off_diag:
        return 1
    else:
        return 2


def switching_score(symmetries):
    T = symmetries.size(0)
    count = 0
    for i in range(T-1):
        if diag_or_not(symmetries[i,:,:]) != diag_or_not(symmetries[i+1,:,:]):
            count = count + 1
    return ((count / T)+0.000000001)

xs = []
ys = []
for i in range(50):
    xs.append((i+1) * (5/50))
    ys.append((switching_score(data0[i,:,:,:])+switching_score(data1[i,:,:,:])+switching_score(data2[i,:,:,:])+switching_score(data3[i,:,:,:]))/4)

np.save('experiment_data1.npy', [xs,ys])
plt.plot(xs, ys)
plt.show()