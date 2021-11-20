"""
X = torch.Tensor([[1, 2, 2], [2, 2, 1], [1, 1, 2]])
ret = torch.Tensor([[70., 82., 82.], [70., 82., 82.], [70., 82., 82.]])
"""

import torch

X = torch.Tensor([[1, 2, 2], [2, 2, 1], [1, 1, 2]])

def cal(X):
    Y = torch.eye(3)
    Y = torch.add(X, Y)
    Z = torch.mul(X, Y)
    A = torch.matmul(Z, X)
    B = torch.sub(A, Y)
    C = torch.sqrt(B)
    D = torch.exp(C)
    return torch.mm(torch.full_like(D, fill_value=2), B)


print(cal(X))
