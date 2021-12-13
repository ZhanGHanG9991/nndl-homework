"""
x = torch.randn((10, 150)).unsqueeze(0)

ret = torch.tensor([[[0.0969, 0.0974, 0.0995],
                     [0.1096, 0.1135, 0.0815],
                     [0.0819, 0.0829, 0.1259],
                     [0.1083, 0.1137, 0.0821],
                     [0.1153, 0.1130, 0.0791],
                     [0.1174, 0.1128, 0.0783],
                     [0.0754, 0.0757, 0.1427],
                     [0.0793, 0.0755, 0.1384],
                     [0.1160, 0.1129, 0.0789],
                     [0.0999, 0.1027, 0.0936]]])
"""

import torch
import torch.nn as nn

x = torch.randn((10, 150)).unsqueeze(0)

ret = torch.tensor([[[0.0969, 0.0974, 0.0995],
                     [0.1096, 0.1135, 0.0815],
                     [0.0819, 0.0829, 0.1259],
                     [0.1083, 0.1137, 0.0821],
                     [0.1153, 0.1130, 0.0791],
                     [0.1174, 0.1128, 0.0783],
                     [0.0754, 0.0757, 0.1427],
                     [0.0793, 0.0755, 0.1384],
                     [0.1160, 0.1129, 0.0789],
                     [0.0999, 0.1027, 0.0936]]])


class Seq_Model(nn.Module):
    def __init__(self,
                 input_size=150,
                 hidden_size=16,
                 num_layers=1,
                 num_class=3):
        super(Seq_Model, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_size * 2,
                                   hidden_size=1,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.drop = torch.nn.Dropout(0.3)
        self.fc1 = nn.Sequential(nn.Linear(2, num_class), nn.Softmax(1))

    def forward(self, x):
        output, (h_n, h_c) = self.lstm1(x, None)
        output, (h_n, h_c) = self.lstm2(output, None)
        ret = self.drop(output)
        return self.fc1(ret)

model = Seq_Model()
print(model.forward(x))