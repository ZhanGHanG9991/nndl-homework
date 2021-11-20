"""
x = torch.randn((64, 64)).unsqueeze(0)
ret = torch.tensor([[[ 6.8746e-02, -8.7003e-02,  1.2126e-01,  2.5498e-01,  2.2506e-01,
          -1.3463e-01],
         [ 6.1387e-02, -1.4329e-01,  6.3237e-02,  9.3049e-02,  3.0875e-02,
          -2.5845e-01],
         [ 7.8355e-02,  7.0930e-03, -1.7036e-01,  2.5199e-01,  5.5748e-02,
          -3.3603e-01],
         [-1.7683e-02,  5.4231e-02, -1.4846e-01,  2.6034e-01, -1.2341e-01,
          -3.6036e-01],
         [ 5.6394e-03,  1.0855e-02, -6.8781e-02,  3.8149e-01,  1.7993e-01,
          -1.2490e-01],
         [-1.9573e-02, -1.3406e-01, -8.8524e-02,  3.2172e-01,  1.8814e-01,
          -2.5965e-02],
         [-6.6759e-05, -7.9932e-02, -1.1594e-01,  2.0263e-01,  1.6774e-01,
          -2.1692e-01],
         [ 2.2534e-04, -2.8178e-02, -3.1305e-02,  1.4228e-01,  8.6752e-02,
          -1.5259e-01],
         [-7.6372e-02, -5.1156e-02,  3.8793e-02,  1.7832e-01,  3.2804e-01,
          -1.3516e-01],
         [ 1.2735e-01, -4.6681e-02,  6.1863e-02,  6.3165e-02,  2.4946e-01,
          -1.5794e-02],
         [ 2.0960e-01, -1.7062e-01,  1.5256e-01,  3.4235e-02,  6.6295e-02,
           4.3568e-02],
         [ 8.6174e-02, -1.5203e-01, -2.7302e-02,  8.0128e-02,  8.9245e-02,
          -1.5984e-01],
         [-1.1698e-02, -9.0143e-03, -1.0140e-01,  1.1772e-01,  2.0376e-02,
          -1.3035e-01],
         [-1.7573e-01, -7.4701e-02, -1.8898e-03,  5.2224e-02,  2.3772e-01,
          -3.0422e-02],
         [-5.2634e-02, -1.3133e-01,  7.0594e-02,  8.2762e-02,  1.6155e-01,
          -1.0713e-01],
         [-6.8292e-02,  1.0931e-01,  2.5579e-02,  1.8657e-01,  1.4379e-01,
          -1.2478e-01],
         [-5.5880e-04, -1.1385e-01,  1.4563e-01,  1.8638e-01,  2.7780e-01,
          -9.8393e-02],
         [ 6.1415e-02, -8.0981e-02,  6.2577e-02,  1.9581e-01,  2.3109e-01,
          -7.2698e-02],
         [ 5.0600e-02, -2.9718e-01,  8.5335e-02,  1.1424e-01,  1.9445e-01,
           1.8953e-02],
         [-3.7678e-02, -9.7197e-02,  8.4173e-02,  6.2759e-02,  1.0282e-02,
           2.6604e-02],
         [-8.3005e-02, -2.1178e-01,  4.7166e-02,  8.3294e-02,  1.9996e-01,
          -2.4229e-01],
         [ 3.7316e-02, -2.9068e-02,  1.3589e-01,  7.9852e-02,  1.7509e-01,
          -1.9008e-01],
         [ 3.2043e-02, -7.1145e-02,  1.1173e-01, -2.5754e-02,  4.1381e-02,
          -3.3565e-01],
         [ 5.1917e-02, -1.1255e-01,  9.4064e-02, -1.1693e-01, -1.0997e-01,
          -2.5160e-01],
         [ 1.1956e-01, -2.0046e-01,  1.1138e-01, -9.1032e-02, -1.0893e-01,
          -2.4597e-01],
         [ 8.8666e-02, -1.7575e-01, -3.9281e-02,  1.1620e-01,  2.5103e-03,
          -1.0202e-01],
         [ 9.2538e-02, -1.2020e-01,  5.1583e-02, -8.7201e-02,  4.3030e-02,
          -1.4445e-02],
         [-4.9177e-02, -1.8027e-02, -1.4465e-01,  3.7262e-02,  1.2565e-01,
          -6.0644e-02],
         [ 4.4669e-02,  4.4801e-02, -1.3655e-01,  6.0806e-02, -6.2787e-02,
          -1.1283e-01],
         [ 5.2075e-02, -3.0120e-01, -9.8786e-02,  3.4985e-02,  8.9500e-02,
          -4.7956e-02],
         [-7.5673e-02, -9.5844e-02, -1.5219e-01,  3.6526e-02, -3.3823e-02,
          -2.6454e-01],
         [-6.8798e-02, -4.8248e-02, -5.7905e-02,  5.8063e-02,  1.1725e-01,
          -6.8112e-02],
         [ 3.5730e-02, -2.5628e-03,  6.3001e-02,  1.4553e-01,  1.9401e-01,
          -4.4992e-02],
         [-4.2532e-02, -1.4846e-01,  4.7157e-02,  9.0109e-02,  2.1980e-01,
          -1.6924e-01],
         [-3.7782e-02,  4.0628e-02,  7.9531e-03,  4.9258e-02,  6.3998e-02,
          -3.5934e-03],
         [ 1.0347e-01, -1.0869e-01, -4.2316e-02,  9.5569e-02,  4.2953e-03,
          -3.6648e-02],
         [ 7.6937e-02, -8.9133e-02, -1.0884e-01,  6.1173e-02,  7.3160e-02,
          -2.2075e-01],
         [-1.5405e-02, -1.9532e-01, -1.2391e-02, -6.7421e-02, -4.7986e-02,
          -4.2265e-02],
         [-1.0651e-01, -1.1743e-01, -1.1023e-01,  3.2355e-02,  1.6952e-02,
          -9.3253e-02],
         [-2.5035e-02, -8.4544e-02, -1.3053e-01,  1.6865e-02,  5.2829e-02,
          -6.2157e-02],
         [-8.2984e-02, -1.7400e-01,  5.2725e-02,  2.1436e-01,  1.3208e-01,
          -3.6419e-02],
         [-1.3676e-01,  7.8320e-02,  4.5608e-02,  2.9969e-01,  8.3817e-02,
          -2.3321e-02],
         [-1.4375e-01,  1.3161e-02,  4.7136e-02,  2.5755e-01,  4.1089e-02,
          -1.3036e-01],
         [-1.4762e-02,  7.0459e-03,  1.2562e-01,  1.9154e-01,  1.1814e-01,
          -6.2922e-02],
         [ 1.7129e-02, -8.2221e-02,  9.9236e-03,  1.5891e-01,  7.9144e-02,
          -9.3347e-02],
         [ 4.2794e-02,  4.0272e-02, -9.3642e-02,  1.3449e-01,  1.6442e-01,
          -1.0782e-01],
         [-1.1169e-01,  1.5768e-01, -2.7603e-01,  1.8358e-01,  1.1481e-01,
          -1.4580e-01],
         [-1.2045e-01, -6.4164e-02, -1.1751e-01,  9.9013e-02, -3.0485e-02,
          -1.5633e-01],
         [ 6.1629e-03, -1.6669e-01, -1.8048e-02,  4.1613e-02, -3.4374e-02,
          -7.5863e-02],
         [ 1.3159e-01, -1.4651e-01, -3.7307e-02, -4.2454e-02,  5.6381e-03,
          -1.5005e-01],
         [-5.4534e-02, -7.8102e-02,  1.7418e-02, -3.9640e-02,  6.0088e-02,
          -2.0966e-01],
         [-1.4550e-01, -1.5528e-02, -2.5055e-04,  7.2513e-02,  1.1681e-01,
          -1.4102e-01],
         [ 4.8053e-02, -4.1802e-02,  8.5783e-02,  8.7869e-02,  1.3409e-01,
          -2.1780e-02],
         [-1.8788e-02, -7.6530e-02,  2.3558e-02,  9.4956e-02,  2.9685e-01,
          -1.0631e-01],
         [-7.3778e-02, -9.6983e-02, -1.0507e-01,  6.5113e-02,  1.1338e-01,
          -5.7298e-02],
         [-1.2245e-02, -7.3088e-02, -6.9710e-02,  6.7439e-02,  1.6660e-01,
          -1.4492e-01],
         [ 2.2434e-02, -3.7892e-02, -5.9810e-02,  6.8218e-02,  9.9003e-02,
          -1.6629e-01],
         [-3.3470e-02,  5.7847e-03,  1.3907e-02,  6.3778e-02,  5.9573e-02,
          -7.0194e-02],
         [-1.0308e-02, -5.1345e-02, -5.5108e-02,  7.2562e-02,  1.5184e-01,
          -1.2956e-01],
         [-1.2007e-02, -1.5278e-01, -7.7234e-02,  1.2279e-01,  1.0129e-01,
          -1.1874e-01],
         [ 1.5292e-02, -9.0743e-02, -1.3307e-01,  8.2755e-02, -3.5366e-02,
          -1.6523e-01],
         [-3.7068e-02, -6.3978e-02,  1.1695e-02,  2.0068e-01,  6.2098e-02,
          -1.8444e-01],
         [ 4.9514e-02, -7.1706e-02, -1.3661e-02,  2.9699e-01, -2.5011e-02,
          -1.9161e-01],
         [ 2.6403e-02, -4.8842e-02, -2.9477e-02,  7.7358e-02, -1.8625e-01,
          -5.1341e-02]]])
"""

import torch
import torch.nn as nn

class RNN_Model(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, num_layers=1, num_class=6):
        super(RNN_Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size * 2, num_class))

    def forward(self, x):
        output, (h_n, h_c) = self.lstm(x, None)
        ret = self.fc1(output)
        return ret