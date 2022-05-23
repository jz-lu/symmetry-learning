import torch
import numpy as np
#data = torch.load('regularizer_data.pt')

data = np.load('experiment_data1.npy')
print(np.shape(data))

data = torch.load('NM_data1.pt')
print(data.size())
print(data[3,33,:,:])

"""
print(data[0,0,3,:,:])
def matrix(p):
    #p[0,:] = torch.tensor([3.14159, 0,3.1415926])
    u0 = torch.tensor([[torch.cos(p[0,0]/2), -torch.exp(-p[0,2]*1j)*torch.sin(p[0,0]/2)],\
                    [torch.exp(p[0,1]*1j)*torch.sin(p[0,0]/2), torch.exp((p[0,1] + p[0,2])*1j)*torch.cos(p[0,0]/2)]])
    u1 = torch.tensor([[torch.cos(p[1,0]/2), -torch.exp(-p[1,2]*1j)*torch.sin(p[1,0]/2)],\
                    [torch.exp(p[1,1]*1j)*torch.sin(p[1,0]/2), torch.exp((p[1,1] + p[1,2])*1j)*torch.cos(p[1,0]/2)]])
    u2 = torch.tensor([[torch.cos(p[2,0]/2), -torch.exp(-p[2,2]*1j)*torch.sin(p[2,0]/2)],\
                    [torch.exp(p[2,1]*1j)*torch.sin(p[2,0]/2), torch.exp((p[2,1] + p[2,2])*1j)*torch.cos(p[2,0]/2)]])
    return [u0, u1, u2]

print(matrix(data[0,0,3,:,:]))
"""