"""
x = torch.Tensor([[0.6798, 0.7467, 0.1265, 0.2276, 0.7954, 0.8569],
                  [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                  [0.6578, 0.7946, 0.5434, 0.6757, 0.7973, 0.9898]])
ret = torch.tensor([[0.5642, 0.5636, 0.4264, 0.4307],
                    [0.5645, 0.5634, 0.4267, 0.4309],
                    [0.5644, 0.5634, 0.4267, 0.4309]])
"""
import torch
import torch.nn as nn


class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(6, 6), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(6, 8), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(8, 12), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(12, 8), nn.Softmax(dim=-1))
        self.layer5 = nn.Sequential(nn.Linear(8, 4), nn.Sigmoid())

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        return layer5_out

x = torch.Tensor([[0.6798, 0.7467, 0.1265, 0.2276, 0.7954, 0.8569],
                  [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                  [0.6578, 0.7946, 0.5434, 0.6757, 0.7973, 0.9898]])
model = Linear_Model()
print(model.forward(x))
