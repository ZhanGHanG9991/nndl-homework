"""
x = torch.randn((16, 128)).unsqueeze(0)
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

x = torch.randn((16, 128)).unsqueeze(0)
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


class Seq_Model(nn.Module):
    def __init__(self,
                 input_size=128,
                 hidden_size=32,
                 num_layers=1,
                 num_class=6):
        super(Seq_Model, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_size * 2,
                                   hidden_size=16,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.lstm3 = torch.nn.LSTM(input_size=32,
                                   hidden_size=8,
                                   num_layers=num_layers,
                                   batch_first=True,
                                   bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(16, num_class), nn.Softmax(1))

    def forward(self, x):
        output, (h_n, h_c) = self.lstm1(x, None)
        output, (h_n, h_c) = self.lstm2(output, None)
        output, (h_n, h_c) = self.lstm3(output, None)
        ret = self.fc1(output)
        return ret
