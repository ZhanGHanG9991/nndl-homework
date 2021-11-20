"""
X_train = torch.tensor([[0.9, 0.1], [1.0, 1.0], [1.0, 2.2], [0.9, 2.4], [0.8, 1.4],
                        [1.4, 0.4], [1.6, 0.4], [1.7, 0.1], [1.9, 1.0], [0.7, 1.1],
                        [3.6, 6.5], [3.7, 6.0], [4.6, 6.9], [4.4, 6.9], [4.1, 5.5],
                        [4.8, 6.4], [4.1, 6.5], [3.5, 7.5], [3.9, 7.0], [3.4, 5.7],
                        [6.7, 2.1], [6.9, 2.1], [5.8, 2.7], [6.8, 3.2], [6.7, 3.3],
                        [6.7, 2.0], [6.3, 2.5], [6.5, 2.0], [6.2, 3.4], [5.9, 3.0]])
y_train = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
ret = torch.tensor([[-0.4900, -0.2600,  1.2000],
                    [-0.7900,  1.0100, -1.9000],
                    [ 0.3900, -0.2700, -0.8000]])
"""

import torch

X_train = torch.tensor([[0.9, 0.1], [1.0, 1.0], [1.0, 2.2], [0.9, 2.4], [0.8, 1.4],
                        [1.4, 0.4], [1.6, 0.4], [1.7, 0.1], [1.9, 1.0], [0.7, 1.1],
                        [3.6, 6.5], [3.7, 6.0], [4.6, 6.9], [4.4, 6.9], [4.1, 5.5],
                        [4.8, 6.4], [4.1, 6.5], [3.5, 7.5], [3.9, 7.0], [3.4, 5.7],
                        [6.7, 2.1], [6.9, 2.1], [5.8, 2.7], [6.8, 3.2], [6.7, 3.3],
                        [6.7, 2.0], [6.3, 2.5], [6.5, 2.0], [6.2, 3.4], [5.9, 3.0]])
y_train = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


def classification_mul(X_train, y_train):

    class Perceptron_Model:
        def __init__(self):
            self.w = torch.ones((3, len(X_train[0])), dtype=torch.float32)
            self.b = torch.tensor([0.0] * 3)
            self.l_rate = 0.1

        def sign(self, x, w, b):
            return torch.dot(x, w) + b

        def fit(self, X_train, Y_train):
            for i in range(3):
                y_train = Y_train.clone().detach()
                if i == 0:
                    for j in range(len(y_train)):
                        if y_train[j] == 0:
                            y_train[j] = 1
                        else:
                            y_train[j] = -1
                elif i == 1:
                    for j in range(len(y_train)):
                        if y_train[j] != 1:
                            y_train[j] = -1
                elif i == 2:
                    for j in range(len(y_train)):
                        if y_train[j] == 2:
                            y_train[j] = 1
                        else:
                            y_train[j] = -1
                is_wrong = False
                while not is_wrong:
                    flag = True
                    for j in range(len(X_train)):
                        s = self.sign(X_train[j], self.w[i], self.b[i])
                        if y_train[j] * s <= 0:
                            flag = False
                            self.w[i] = self.w[i] + self.l_rate * y_train[j] * X_train[j]
                            self.b[i] = self.b[i] + self.l_rate * y_train[j]
                    if flag:
                        is_wrong = True

    model = Perceptron_Model()
    model.fit(X_train, y_train)
    return torch.hstack((model.w, torch.reshape(model.b, (3, 1))))


print(classification_mul(X_train, y_train))
