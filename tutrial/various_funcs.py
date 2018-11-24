import torch
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(y[0].item())

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)