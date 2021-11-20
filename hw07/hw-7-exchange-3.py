"""
X = torch.tensor([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]])
ret = torch.tensor([[14, 20, 26],
                    [20, 29, 38],
                    [26, 38, 50]])
"""
import torch


def func(X):
    Y = X.T
    return torch.mm(X, Y)


print(func(X))
