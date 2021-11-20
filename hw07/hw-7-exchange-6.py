"""
X = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
ret = torch.tensor([[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]])
"""
import torch


def func(X):
    return X.view((3, 3))
