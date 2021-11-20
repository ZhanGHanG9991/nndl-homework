"""
X = torch.Tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
Y = torch.Tensor([[1, 2, 3], [7, 8, 9]])
ret = torch.Tensor([0, 0, 0, 2, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0])
"""

import torch

def cat01(X, Y):
    x_ones = torch.ones_like(X)
    x_rand = torch.rand_like(X)
    x_zero = torch.zeros_like(X)
    y_shape = Y.shape
    rand_tensor = torch.rand(y_shape)
    ones_tensor = torch.ones(y_shape)
    zeros_tensor = torch.zeros(y_shape)
    cat_tensor = torch.cat([x_ones, x_rand, x_zero, rand_tensor, ones_tensor, zeros_tensor], dim=0)
    sort_tensor = torch.sort(cat_tensor, dim=1)
    index_sort_tensor = torch.min(sort_tensor[1], dim=1)[1]
    return index_sort_tensor