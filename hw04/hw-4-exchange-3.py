"""
X = torch.Tensor([-1.0, -1.2, 2.0, 3.0, 2.0, 1.0, 1.0, -8.6])
ret = torch.tensor([1., 2., 2., 3., 2., 1., 1., 9.])
"""

import torch

def cat03(X):
    return torch.ceil(torch.abs(X))

