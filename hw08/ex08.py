"""
x = torch.randn((10, 256)).unsqueeze(0)
ret = torch.tensor([[[0.0600, 0.0631, 0.0617, 0.0632, 0.0619, 0.0626],
                     [0.0607, 0.0620, 0.0617, 0.0626, 0.0622, 0.0641],
                     [0.0617, 0.0611, 0.0618, 0.0618, 0.0624, 0.0643],
                     [0.0632, 0.0610, 0.0621, 0.0622, 0.0620, 0.0639],
                     [0.0634, 0.0614, 0.0624, 0.0633, 0.0628, 0.0636],
                     [0.0629, 0.0624, 0.0622, 0.0637, 0.0634, 0.0630],
                     [0.0623, 0.0631, 0.0616, 0.0643, 0.0635, 0.0633],
                     [0.0619, 0.0636, 0.0619, 0.0648, 0.0632, 0.0625],
                     [0.0613, 0.0624, 0.0616, 0.0663, 0.0615, 0.0625],
                     [0.0614, 0.0620, 0.0616, 0.0659, 0.0603, 0.0624],
                     [0.0621, 0.0625, 0.0620, 0.0641, 0.0606, 0.0622],
                     [0.0627, 0.0628, 0.0624, 0.0620, 0.0619, 0.0616],
                     [0.0633, 0.0628, 0.0633, 0.0605, 0.0633, 0.0608],
                     [0.0642, 0.0634, 0.0637, 0.0592, 0.0635, 0.0608],
                     [0.0648, 0.0637, 0.0645, 0.0587, 0.0635, 0.0606],
                     [0.0642, 0.0627, 0.0654, 0.0573, 0.0643, 0.0618]]])
"""

import torch
import torch.nn as nn


class Seq_Model(nn.Module):
    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 num_layers=2,
                 num_class=7):
        super(Seq_Model, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, num_class), nn.ReLU(1))
        self.fc2 = nn.Sequential(nn.Linear(num_class, num_class), nn.Softmax(1))

    def forward(self, x):
        output, (h_n, h_c) = self.lstm1(x, None)
        output = self.fc1(output)
        output = self.fc2(output)
        return ret

model = Seq_Model()
print(model.forward(x))
