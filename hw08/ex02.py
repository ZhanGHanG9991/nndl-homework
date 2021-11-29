"""
x = torch.randn((10, 256)).unsqueeze(0)
ret = torch.tensor([[[0.0927, 0.1042, 0.0915, 0.1050, 0.1077, 0.1033, 0.0900],
                     [0.0992, 0.1011, 0.0991, 0.1037, 0.0996, 0.1037, 0.0974],
                     [0.1015, 0.0995, 0.1010, 0.1014, 0.0979, 0.1033, 0.0998],
                     [0.1017, 0.0990, 0.1012, 0.0998, 0.0980, 0.1024, 0.1006],
                     [0.1013, 0.0994, 0.1015, 0.0993, 0.0985, 0.1005, 0.1013],
                     [0.1017, 0.0989, 0.1021, 0.0981, 0.0986, 0.0993, 0.1022],
                     [0.1014, 0.0992, 0.1016, 0.0979, 0.0990, 0.0980, 0.1024],
                     [0.1010, 0.0992, 0.1009, 0.0976, 0.0996, 0.0973, 0.1024],
                     [0.1002, 0.0995, 0.1005, 0.0980, 0.1003, 0.0964, 0.1022],
                     [0.0993, 0.1000, 0.1006, 0.0992, 0.1009, 0.0959, 0.1018]]])
"""

import torch
import torch.nn as nn

class Seq_Model(nn.Module):
    def __init__(self, input_size=256, hidden_size=32, num_layers=2, num_class=7):
        super(Seq_Model, self).__init__()
        self.lstm=torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True, bidirectional=True)
        self.gru=torch.nn.GRU(input_size=hidden_size * 2, hidden_size=num_class, num_layers=num_layers,
                                 batch_first=True)
        self.fc1=nn.Sequential(nn.Linear(num_class, num_class), nn.Softmax(1))

    def forward(self, x):
        output, (h_n, h_c)=self.lstm(x, None)
        output, (h_n, h_c)=self.gru(output, None)
        ret=self.fc1(output)
        return ret