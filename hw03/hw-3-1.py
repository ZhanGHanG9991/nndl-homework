"""
x = torch.tensor([[0.6798, 0.3467, 0.1265, 0.2276, 0.4954, 0.3569],
                  [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                  [0.1535, 0.7946, 0.1434, 0.6757, 0.7973, 0.4898]])
ret = torch.tensor([[0.5039, 0.5265],
                    [0.5454, 0.5327],
                    [0.5082, 0.5094]])
"""
import torch
import torch.nn as nn

x = torch.Tensor([[0.6798, 0.3467, 0.1265, 0.2276, 0.4954, 0.3569],
                  [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                  [0.1535, 0.7946, 0.1434, 0.6757, 0.7973, 0.4898]])


class Linear_Model(nn.Module):
    def __init__(self, mid_dim=10):
        super(Linear_Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(6, mid_dim), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(mid_dim, 2), nn.Sigmoid())

    def forward(self, x):
        mid_out = self.layer1(x)
        ret = self.layer2(mid_out)
        return ret


model = Linear_Model()
print(model.forward(x))
