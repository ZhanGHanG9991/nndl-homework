"""
X = torch.Tensor([[[1, 2], [2, 1], [3, 2]]])
Y = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
ret = torch.Tensor([[1., 3., 2., 8.], [1., 3., 2., 0.], [1., 2., 0., 0.]])
"""

import torch

X = torch.Tensor([[[1, 2], [2, 1], [3, 2]]])
Y = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])

def operate(X, Y):
    X = torch.squeeze(X, 0)
    X = torch.reshape(X, (2, -1))
    Z = torch.cat([X, torch.reshape(torch.tensor([torch.numel(X), torch.numel(Y)]), (2, -1))], dim=1)
    Z = torch.unsqueeze(Z, 0)
    A = torch.cat([Z, Z], dim=0)
    B = torch.cat([A[0], A[1]], dim=0)
    return torch.topk(torch.tril(B), 3, dim=0)[0]


print(operate(X, Y))
