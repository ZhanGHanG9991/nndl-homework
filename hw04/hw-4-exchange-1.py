"""
X = torch.Tensor([4.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
Y = torch.Tensor([2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0])
ret = torch.tensor(4.3589)
"""

import torch

X = torch.Tensor([4.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
Y = torch.Tensor([2.0, 2.0, 2.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0])


def cat01(X, Y):
    return torch.sqrt(torch.sum(torch.mul((X - Y), (X - Y))))

cat01(X, Y)
