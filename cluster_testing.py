for i in range(10):
    print(i)

import torch

a = torch.ones(3,3)
torch.save(a, 'testing_tensor.pt')