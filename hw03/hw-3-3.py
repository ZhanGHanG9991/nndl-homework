"""
x1 = torch.Tensor([[0.6798, 0.7467, 0.1265, 0.2276, 0.7954, 0.8569],
                   [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                   [0.6578, 0.7946, 0.5434, 0.6757, 0.7973, 0.9898]])

x2 = torch.Tensor([[0.1238, 0.4767, 0.3765, 0.8756, 0.7954, 0.2541, 0.8736, 0.2365],
                   [0.3656, 0.9981, 0.5871, 0.8897, 0.2561, 0.1329, 0.9856, 0.7654],
                   [0.4441, 0.5766, 0.9804, 0.6751, 0.5763, 0.1098, 0.7632, 0.9853]])

ret = torch.tensor([[0.1452, 0.1604, 0.1591, 0.1224, 0.1190, 0.0750, 0.1503, 0.0687],
                    [0.1452, 0.1605, 0.1591, 0.1224, 0.1189, 0.0749, 0.1505, 0.0686],
                    [0.1451, 0.1604, 0.1591, 0.1223, 0.1190, 0.0749, 0.1505, 0.0687]])
"""

import torch
import torch.nn as nn

x1 = torch.Tensor([[0.6798, 0.7467, 0.1265, 0.2276, 0.7954, 0.8569],
                   [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                   [0.6578, 0.7946, 0.5434, 0.6757, 0.7973, 0.9898]])

x2 = torch.Tensor([[0.1238, 0.4767, 0.3765, 0.8756, 0.7954, 0.2541, 0.8736, 0.2365],
                   [0.3656, 0.9981, 0.5871, 0.8897, 0.2561, 0.1329, 0.9856, 0.7654],
                   [0.4441, 0.5766, 0.9804, 0.6751, 0.5763, 0.1098, 0.7632, 0.9853]])


class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()

        self.layer1_1 = nn.Sequential(nn.Linear(6, 8), nn.ReLU())
        self.layer1_2 = nn.Sequential(nn.Linear(8, 12), nn.ReLU())
        self.layer1_3 = nn.Sequential(nn.Linear(12, 16), nn.Softmax(dim=-1))
        self.layer1_4 = nn.Sequential(nn.Linear(16, 20), nn.ReLU())

        self.layer2_1 = nn.Sequential(nn.Linear(8, 10), nn.ReLU())
        self.layer2_2 = nn.Sequential(nn.Linear(10, 12), nn.Tanh())
        self.layer2_3 = nn.Sequential(nn.Linear(12, 20), nn.ReLU())
        self.layer2_4 = nn.Sequential(nn.Linear(20, 12), nn.Sigmoid())

        self.layer3_1 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.layer3_2 = nn.Sequential(nn.Linear(16, 8), nn.Softmax(dim=-1))

    def forward(self, x1, x2):
        x1_1 = self.layer1_1(x1)
        x1_2 = self.layer1_2(x1_1)
        x1_3 = self.layer1_3(x1_2)
        x1_4 = self.layer1_4(x1_3)

        x2_1 = self.layer2_1(x2)
        x2_2 = self.layer2_2(x2_1)
        x2_3 = self.layer2_3(x2_2)
        x2_4 = self.layer2_4(x2_3)

        x = torch.cat((x1_4, x2_4), dim=-1)
        x3_1 = self.layer3_1(x)
        x3_2 = self.layer3_2(x3_1)
        return x3_2

model = Linear_Model()
print(model.forward(x1, x2))
