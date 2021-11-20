"""
output = torch.Tensor([1.2, 0.3, 0.6])
target = torch.Tensor([1, 0, 1])
ret = torch.tensor(0.0967)
"""

import torch

def squared_mean_loss(output, target):
    sub = torch.sub(output, target)
    square = torch.mul(sub, sub)
    return torch.mean(square)

