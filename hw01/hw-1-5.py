"""
X = torch.Tensor([1, 2, 3])
ret = torch.tensor([[ 1.0000,  0.8415,  0.6664, -2.1188,  0.0000,  0.5000],
                    [ 1.0000,  0.8415,  0.6664,  0.0635,  0.0000,  0.5000],
                    [ 2.0000,  0.9093,  0.6143, -1.4555,  0.6931,  0.6667],
                    [ 2.0000,  0.9093,  0.6143, -0.0126,  0.6931,  0.6667],
                    [ 3.0000,  0.1411,  0.9901, -0.1548,  1.0986,  0.7500],
                    [ 3.0000,  0.1411,  0.9901, -0.0927,  1.0986,  0.7500]])
"""

import torch

X = torch.Tensor([1, 2, 3])

def func01(X):
    torch.manual_seed(666)
    Y = torch.repeat_interleave(X, 2)
    Y = torch.unsqueeze(Y, 0)
    A = torch.sin(Y)
    B = torch.cos(A)
    C = torch.normal(0, 1, (1, 6))
    D = torch.log(Y)
    D = torch.abs(D)
    E = torch.sigmoid(D)
    square = torch.cat([Y, A, B, C, D, E], dim=0)
    return torch.transpose(square, dim0=0, dim1=1)


print(func01(X))
