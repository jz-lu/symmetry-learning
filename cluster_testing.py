

import torch
from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

data = torch.load('regularizer_data.pt')
#print(data[0,0,0,:,:])
print(data.size())


for i in range(3):
    torch.save(data, 'test.pt')



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
    T = torch.size(0)
    count = 0
    for i in range(T-1):
        if diag_or_not(symmetries[i,:,:]) != diag_or_not(symmetries[i+1,:,:]):
            count = count + 1
    return count / T

def plotting_function(x,y):
    i = np.round(x * 10 - 1)
    j = np.round(y * 100 - 1)
    print(i,j)
    return switching_score(data[i,j,:,:,:])

x = np.linspace(0.1, 1, 10)
y = np.linspace(0.01, 0.1, 10)

X, Y = np.meshgrid(x, y)
Z = plotting_function(X, Y)    
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
