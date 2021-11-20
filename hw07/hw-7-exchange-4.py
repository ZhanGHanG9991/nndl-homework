"""
X = torch.tensor([1,2,3])
Y = torch.tensor([2,3,4])
ret = torch.tensor([1,2,3])
"""
import torch


def func(X, Y):
    Z = X + Y
    A = X - Y
    mi = torch.min(X)
    ma = torch.max(Y)
    return X
