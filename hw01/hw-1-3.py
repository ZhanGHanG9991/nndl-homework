"""
X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
ret = torch.tensor([5.0000e+00, 3.6288e+05, 2.7386e+00, 4.5000e+01, 7.5000e+00, 1.0000e+00, 1.0000e+00, 9.0000e+00], dtype=torch.float64)
"""

import torch

def agg01(X):
    Y = torch.eye(X.shape[0])
    l = []
    l.append(torch.mean(X))
    l.append(torch.prod(X))
    l.append(torch.std(X))
    l.append(torch.sum(X))
    l.append(torch.var(X))
    l.append(torch.tensor(1 if torch.equal(X, torch.mm(X, Y)) else 0))
    l.append(torch.min(X))
    l.append(torch.max(X))
    return torch.stack(l)

print(agg01(X))
