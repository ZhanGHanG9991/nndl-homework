"""
output = torch.Tensor([2.0, 1.3, 1.6])
target = torch.Tensor([1, 0, 1])
ret = torch.tensor(0.0967)
"""

import torch

def smloss(output, target):
    sub = torch.sub(output, target)
    square = torch.mul(sub, sub)
    return torch.mean(square)