"""
X = torch.Tensor([0.7, 0.3, 0.6])
Y = torch.Tensor([0.9, 0.2, 0.8])
ret = torch.tensor(0.5198)
"""

import torch

X = torch.Tensor([0.7, 0.3, 0.6])
Y = torch.Tensor([0.9, 0.2, 0.8])

def cross_entropy_mean_loss(output, target):

    sub = torch.sub(torch.mul(-target, torch.log(output)), torch.mul((1 - target), torch.log(1 - output)))
    return sub.mean()

print(cross_entropy_mean_loss(X, Y))
