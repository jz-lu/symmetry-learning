import torch
import numpy as np
import matplotlib.pyplot as plt

#data = torch.load('regularizer_data.pt')

data = torch.load('NM_data1.pt')
print(data.size())


def matrix(p):
    #p[0,:] = torch.tensor([3.14159, 0,3.1415926])
    u0 = torch.tensor([[torch.cos(p[0,0]/2), -torch.exp(-p[0,2]*1j)*torch.sin(p[0,0]/2)],\
                    [torch.exp(p[0,1]*1j)*torch.sin(p[0,0]/2), torch.exp((p[0,1] + p[0,2])*1j)*torch.cos(p[0,0]/2)]])
    u1 = torch.tensor([[torch.cos(p[1,0]/2), -torch.exp(-p[1,2]*1j)*torch.sin(p[1,0]/2)],\
                    [torch.exp(p[1,1]*1j)*torch.sin(p[1,0]/2), torch.exp((p[1,1] + p[1,2])*1j)*torch.cos(p[1,0]/2)]])
    u2 = torch.tensor([[torch.cos(p[2,0]/2), -torch.exp(-p[2,2]*1j)*torch.sin(p[2,0]/2)],\
                    [torch.exp(p[2,1]*1j)*torch.sin(p[2,0]/2), torch.exp((p[2,1] + p[2,2])*1j)*torch.cos(p[2,0]/2)]])
    return [u0, u1, u2]


for i in range(100):
    print(matrix(data[40,i,:,:]))

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
    xs.append((i+1) * (1/50))
    ys.append(switching_score(data[i,:,:,:]))


plt.plot(xs, ys)
plt.show()