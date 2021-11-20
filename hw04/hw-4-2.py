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

ret = torch.tensor([[ 1.3362, -1.0642, -0.6172, -0.6712]])

"""

import torch
import torch.nn as nn


class ResNet_Model(nn.Module):

    def __init__(self, kernel_size=4, stride=1, num_block=5):

        class ResBaseBlock(nn.Module):
            def __init__(self, n_chans):
                super(ResBaseBlock, self).__init__()
                self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
                self.batch_norm = nn.BatchNorm2d(num_features=n_chans)

            def forward(self, x):
                out = self.conv(x)
                out = self.batch_norm(out)
                out = torch.relu(out)
                return out + x

        super(ResNet_Model, self).__init__()
        self.conv = nn.Conv2d(4, 8, kernel_size=kernel_size, stride=stride, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.res_blocks = nn.Sequential(*(num_block * [ResBaseBlock(n_chans=8)]))
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        out = self.conv(x)
        out = self.res_blocks(out)
        out = self.pool(out).squeeze(-1).squeeze(-1)
        ret = self.fc(out)
        return ret


model = ResNet_Model()
print(model.forward(x))
