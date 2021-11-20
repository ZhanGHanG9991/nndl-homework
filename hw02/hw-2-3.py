"""
ret = torch.tensor([ 0.2604,  0.3542,  0.2635, -0.1347])
"""

import torch

def regression():

    len = 2000

    x = torch.linspace(0, 3.14106, len)
    y = torch.sin(x) + (0.005**0.5) * torch.randn(len)

    a = b = c = d = 0
    learning_rate = 1e-6

    for t in range(len):

        y_predict = a + b * x + c * x ** 2 + d * x ** 3

        gradient_y_predict = 2.0 * (y_predict - y)
        gradient_a = gradient_y_predict.sum()
        gradient_b = (gradient_y_predict * x).sum()
        gradient_c = (gradient_y_predict * x ** 2).sum()
        gradient_d = (gradient_y_predict * x ** 3).sum()

        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b
        c -= learning_rate * gradient_c
        d -= learning_rate * gradient_d

    return torch.tensor([a, b, c, d])

print(regression())