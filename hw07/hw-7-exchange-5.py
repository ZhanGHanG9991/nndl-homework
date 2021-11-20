"""
X = torch.tensor([[1, 2, 3]])
ret = torch.tensor([1, 2, 3])
"""
import torch


def func(X):
    return X.squeeze(0)


print(func(X))
