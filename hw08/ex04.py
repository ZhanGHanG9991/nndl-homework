"""
x = torch.randn((16, 50)).unsqueeze(0)
ret = torch.tensor([[[0.0635, 0.0604, 0.0601, 0.0601, 0.0617, 0.0623, 0.0635, 0.0617,
                      0.0667, 0.0621, 0.0616, 0.0610, 0.0627, 0.0632, 0.0628, 0.0649,
                      0.0646, 0.0625, 0.0667, 0.0620, 0.0642, 0.0608, 0.0644, 0.0627,
                      0.0640, 0.0649, 0.0657, 0.0598, 0.0648, 0.0618, 0.0614, 0.0630],
                    [0.0626, 0.0634, 0.0644, 0.0627, 0.0614, 0.0606, 0.0624, 0.0611,
                        0.0657, 0.0608, 0.0620, 0.0586, 0.0658, 0.0611, 0.0629, 0.0623,
                        0.0647, 0.0627, 0.0624, 0.0609, 0.0633, 0.0642, 0.0620, 0.0622,
                        0.0674, 0.0676, 0.0618, 0.0651, 0.0626, 0.0637, 0.0658, 0.0601],
                     [0.0652, 0.0628, 0.0647, 0.0619, 0.0623, 0.0620, 0.0624, 0.0609,
                         0.0614, 0.0617, 0.0626, 0.0641, 0.0638, 0.0636, 0.0608, 0.0611,
                         0.0595, 0.0630, 0.0629, 0.0656, 0.0636, 0.0611, 0.0597, 0.0591,
                         0.0640, 0.0622, 0.0612, 0.0628, 0.0634, 0.0618, 0.0624, 0.0612],
                     [0.0632, 0.0629, 0.0636, 0.0639, 0.0656, 0.0629, 0.0625, 0.0604,
                         0.0625, 0.0576, 0.0633, 0.0625, 0.0642, 0.0633, 0.0609, 0.0638,
                         0.0604, 0.0616, 0.0608, 0.0619, 0.0636, 0.0656, 0.0632, 0.0614,
                         0.0655, 0.0623, 0.0588, 0.0663, 0.0606, 0.0635, 0.0639, 0.0607],
                     [0.0636, 0.0617, 0.0656, 0.0636, 0.0611, 0.0629, 0.0606, 0.0600,
                         0.0599, 0.0593, 0.0590, 0.0629, 0.0624, 0.0629, 0.0627, 0.0608,
                         0.0623, 0.0650, 0.0620, 0.0601, 0.0629, 0.0651, 0.0639, 0.0648,
                         0.0664, 0.0630, 0.0583, 0.0629, 0.0633, 0.0629, 0.0637, 0.0624],
                     [0.0615, 0.0643, 0.0614, 0.0634, 0.0620, 0.0621, 0.0613, 0.0619,
                         0.0607, 0.0636, 0.0613, 0.0636, 0.0606, 0.0645, 0.0634, 0.0607,
                         0.0632, 0.0640, 0.0591, 0.0617, 0.0620, 0.0631, 0.0644, 0.0627,
                         0.0630, 0.0630, 0.0621, 0.0653, 0.0608, 0.0629, 0.0638, 0.0633],
                     [0.0605, 0.0626, 0.0605, 0.0609, 0.0629, 0.0647, 0.0614, 0.0630,
                         0.0614, 0.0652, 0.0637, 0.0648, 0.0609, 0.0624, 0.0641, 0.0609,
                         0.0629, 0.0629, 0.0629, 0.0630, 0.0617, 0.0611, 0.0630, 0.0629,
                         0.0587, 0.0625, 0.0623, 0.0617, 0.0606, 0.0613, 0.0614, 0.0637],
                     [0.0606, 0.0641, 0.0630, 0.0640, 0.0633, 0.0627, 0.0612, 0.0643,
                         0.0617, 0.0641, 0.0653, 0.0614, 0.0614, 0.0621, 0.0658, 0.0612,
                         0.0628, 0.0604, 0.0595, 0.0629, 0.0607, 0.0625, 0.0629, 0.0628,
                         0.0593, 0.0638, 0.0623, 0.0640, 0.0603, 0.0607, 0.0639, 0.0624],
                     [0.0603, 0.0626, 0.0604, 0.0617, 0.0629, 0.0628, 0.0624, 0.0662,
                         0.0677, 0.0658, 0.0662, 0.0607, 0.0607, 0.0613, 0.0648, 0.0618,
                         0.0682, 0.0597, 0.0600, 0.0614, 0.0605, 0.0642, 0.0618, 0.0623,
                         0.0607, 0.0636, 0.0650, 0.0639, 0.0604, 0.0623, 0.0632, 0.0633],
                     [0.0625, 0.0624, 0.0620, 0.0634, 0.0618, 0.0625, 0.0635, 0.0630,
                         0.0655, 0.0651, 0.0642, 0.0614, 0.0600, 0.0626, 0.0656, 0.0623,
                         0.0669, 0.0592, 0.0600, 0.0608, 0.0606, 0.0632, 0.0622, 0.0619,
                         0.0607, 0.0621, 0.0633, 0.0641, 0.0614, 0.0627, 0.0618, 0.0634],
                     [0.0656, 0.0615, 0.0637, 0.0606, 0.0583, 0.0613, 0.0621, 0.0615,
                         0.0605, 0.0611, 0.0601, 0.0646, 0.0624, 0.0639, 0.0625, 0.0591,
                         0.0622, 0.0612, 0.0660, 0.0631, 0.0615, 0.0613, 0.0650, 0.0645,
                         0.0598, 0.0616, 0.0609, 0.0597, 0.0635, 0.0635, 0.0618, 0.0679],
                     [0.0619, 0.0631, 0.0641, 0.0637, 0.0610, 0.0640, 0.0641, 0.0639,
                         0.0621, 0.0615, 0.0643, 0.0624, 0.0630, 0.0612, 0.0606, 0.0624,
                         0.0597, 0.0618, 0.0626, 0.0628, 0.0627, 0.0631, 0.0614, 0.0618,
                         0.0628, 0.0623, 0.0609, 0.0620, 0.0612, 0.0629, 0.0617, 0.0626],
                     [0.0607, 0.0636, 0.0607, 0.0636, 0.0663, 0.0626, 0.0627, 0.0628,
                         0.0605, 0.0636, 0.0626, 0.0639, 0.0641, 0.0615, 0.0621, 0.0647,
                         0.0607, 0.0635, 0.0627, 0.0636, 0.0623, 0.0612, 0.0620, 0.0620,
                         0.0606, 0.0619, 0.0631, 0.0620, 0.0630, 0.0605, 0.0616, 0.0607],
                     [0.0595, 0.0640, 0.0600, 0.0631, 0.0642, 0.0639, 0.0632, 0.0637,
                         0.0610, 0.0630, 0.0631, 0.0631, 0.0625, 0.0613, 0.0606, 0.0649,
                         0.0602, 0.0650, 0.0637, 0.0625, 0.0636, 0.0599, 0.0626, 0.0621,
                         0.0612, 0.0606, 0.0653, 0.0612, 0.0616, 0.0624, 0.0611, 0.0623],
                     [0.0635, 0.0599, 0.0631, 0.0628, 0.0632, 0.0623, 0.0628, 0.0621,
                         0.0613, 0.0605, 0.0597, 0.0617, 0.0616, 0.0620, 0.0601, 0.0652,
                         0.0605, 0.0638, 0.0644, 0.0629, 0.0628, 0.0639, 0.0630, 0.0652,
                         0.0633, 0.0600, 0.0626, 0.0598, 0.0668, 0.0634, 0.0602, 0.0622],
                     [0.0654, 0.0605, 0.0628, 0.0607, 0.0619, 0.0604, 0.0639, 0.0636,
                         0.0615, 0.0651, 0.0609, 0.0634, 0.0640, 0.0628, 0.0604, 0.0639,
                         0.0614, 0.0640, 0.0645, 0.0648, 0.0640, 0.0597, 0.0584, 0.0617,
                         0.0626, 0.0588, 0.0664, 0.0594, 0.0657, 0.0637, 0.0623, 0.0608]]])
"""

import torch
import torch.nn as nn


class Seq_Model(nn.Module):
    def __init__(self,
                 input_size=50,
                 hidden_size=50,
                 num_layers=1,
                 num_class=32):
        super(Seq_Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size * 2, num_class),
                                 nn.ReLU(1))
        self.fc2 = nn.Sequential(nn.Linear(num_class, num_class),
                                 nn.Softmax(1))

    def forward(self, x):
        output, (h_n, h_c) = self.lstm(x, None)
        ret = self.fc1(output)
        return self.fc2(ret)
