"""
x = torch.Tensor([[[[0.6358, 0.3338, 0.4582, 0.1802],
                    [0.3568, 0.6443, 0.5283, 0.7092],
                    [0.4561, 0.0499, 0.8935, 0.3299],
                    [0.3293, 0.3670, 0.4018, 0.8958]],

                   [[0.5686, 0.8445, 0.2931, 0.6898],
                    [0.4569, 0.1537, 0.0330, 0.9367],
                    [0.1050, 0.0841, 0.8211, 0.1595],
                    [0.7024, 0.1042, 0.8950, 0.2350]],

                   [[0.0142, 0.0258, 0.4181, 0.3458],
                    [0.3452, 0.2344, 0.3457, 0.7344],
                    [0.7905, 0.7611, 0.1271, 0.5876],
                    [0.6685, 0.4569, 0.8676, 0.8478]],

                   [[0.1132, 0.7258, 0.5156, 0.9348],
                    [0.5434, 0.3067, 0.6367, 0.4560],
                    [0.6565, 0.5341, 0.7456, 0.3466],
                    [0.7656, 0.8209, 0.9676, 0.3345]]]])

ret = torch.tensor([[-0.0314, -0.0899, -0.3367,  0.3342, -0.1811]])

"""

import torch
import torch.nn as nn

x = torch.Tensor([[[[0.6358, 0.3338, 0.4582, 0.1802],
                    [0.3568, 0.6443, 0.5283, 0.7092],
                    [0.4561, 0.0499, 0.8935, 0.3299],
                    [0.3293, 0.3670, 0.4018, 0.8958]],

                   [[0.5686, 0.8445, 0.2931, 0.6898],
                    [0.4569, 0.1537, 0.0330, 0.9367],
                    [0.1050, 0.0841, 0.8211, 0.1595],
                    [0.7024, 0.1042, 0.8950, 0.2350]],

                   [[0.0142, 0.0258, 0.4181, 0.3458],
                    [0.3452, 0.2344, 0.3457, 0.7344],
                    [0.7905, 0.7611, 0.1271, 0.5876],
                    [0.6685, 0.4569, 0.8676, 0.8478]],

                   [[0.1132, 0.7258, 0.5156, 0.9348],
                    [0.5434, 0.3067, 0.6367, 0.4560],
                    [0.6565, 0.5341, 0.7456, 0.3466],
                    [0.7656, 0.8209, 0.9676, 0.3345]]]])


class CNN_Model(nn.Module):
    def __init__(self, in_dim=4, out_dim=16, kernel_size=4, stride=1, num_class=5):
        super(CNN_Model, self).__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_dim, 8)
        self.fc2 = nn.Linear(8, num_class)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pool(x1).squeeze(-1).squeeze(-1)
        x3 = self.fc1(x2)
        ret = self.fc2(x3)
        return ret


model = CNN_Model()
print(model.forward(x))
