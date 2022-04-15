

import torch
from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt


ran = np.load('experiment_data.npy')
print(ran)

data = torch.load('regularizer_data.pt')
#print(data[0,0,0,:,:])
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

print(switching_score(data[1,1,:,:,:]))

def plotting_function(x,y):
    i = np.round(x * 10 - 1)
    j = np.round(y * 100 - 1)
    print(i,j)
    return switching_score(data[i,j,:,:,:])


xs=[]
ys = []
zs=[]

for t in range(100):
    #print(t, t%10, int(t/10))
    xs.append((t%10)*0.1)
    ys.append(int(t/10)* 0.01)
    zs.append(switching_score(data[t%10, int(t/10),:,:,:]))
    print(t)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xs, ys, zs)

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Finite Difference Step Size')
ax.set_zlabel('Score Function')

plt.show()

np.save('experiment_data.npy', [xs,ys,zs])
