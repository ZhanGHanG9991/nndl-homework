"""
x = torch.Tensor([[[0.6798, 0.3467, 0.1265, 0.2276, 0.4954, 0.3569],
                   [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                   [0.1535, 0.7946, 0.1434, 0.6757, 0.7973, 0.4898],
                   [0.5686, 0.8445, 0.2931, 0.6898, 0.4569, 0.1537],
                   [0.0330, 0.9367, 0.1050, 0.0841, 0.8211, 0.1595]]])

ret = torch.tensor([0.5758, 0.4730, 0.4643])

"""

import torch
import torch.nn as nn

x = torch.Tensor([[[0.6798, 0.3467, 0.1265, 0.2276, 0.4954, 0.3569],
                   [0.1576, 0.4466, 0.5766, 0.5567, 0.2034, 0.0329],
                   [0.1535, 0.7946, 0.1434, 0.6757, 0.7973, 0.4898],
                   [0.5686, 0.8445, 0.2931, 0.6898, 0.4569, 0.1537],
                   [0.0330, 0.9367, 0.1050, 0.0841, 0.8211, 0.1595]]])


class RNN_Model(nn.Module):
    def __init__(self, input_size=6, hidden_size=10, num_layers=1, num_class=3):
        super(RNN_Model, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(5, num_class), nn.Sigmoid())

    def forward(self, x):
        output, (h_n, h_c) = self.rnn(x, None)
        output = torch.mean(output, dim=2).squeeze(0)
        ret = self.fc1(output)
        return ret
model = RNN_Model()
print(model.forward(x))