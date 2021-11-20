"""
X = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
i = 2
ret = torch.tensor([[1, 2, 3],
                    [2, 3, 4],
                    [5, 6, 7]])
"""

import torch


def func(X, i):
    X[i] = torch.add(X[i], i)
    return X
