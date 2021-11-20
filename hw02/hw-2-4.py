"""
X_train = torch.tensor([[4.9, 3.1], [5.0, 3.0], [5.9, 3.2], [4.9, 2.4], [4.8, 3.4],
                        [5.4, 3.4], [4.6, 3.4], [6.7, 3.1], [6.6, 3.0], [6.9, 3.1],
                        [5.6, 2.5], [5.7, 3.0], [6.6, 2.9], [4.4, 2.9], [5.1, 2.5],
                        [4.8, 3.4], [5.1, 3.5], [5.5, 3.5], [5.9, 3.0], [5.4, 3.7]])
y_train = torch.tensor([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1])
ret = torch.tensor([ 1.0200, -1.6400, -0.6000])
"""

import torch

X_train = torch.tensor([[4.9, 3.1], [5.0, 3.0], [5.9, 3.2], [4.9, 2.4], [4.8, 3.4],
                        [5.4, 3.4], [4.6, 3.4], [6.7, 3.1], [6.6, 3.0], [6.9, 3.1],
                        [5.6, 2.5], [5.7, 3.0], [6.6, 2.9], [4.4, 2.9], [5.1, 2.5],
                        [4.8, 3.4], [5.1, 3.5], [5.5, 3.5], [5.9, 3.0], [5.4, 3.7]])
y_train = torch.tensor([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1])


def classification_two(X_train, y_train):

    class Perceptron_Model:
        def __init__(self):
            self.w = torch.ones(len(X_train[0]), dtype=torch.float32)
            self.b = 0
            self.l_rate = 0.1

        def sign(self, x, w, b):
            return torch.dot(x, w) + b

        def fit(self, X_train, y_train):
            is_wrong = False
            while not is_wrong:
                flag = True
                for i in range(len(X_train)):
                    s = self.sign(X_train[i], self.w, self.b)
                    if y_train[i] * s <= 0:
                        flag = False
                        self.w = self.w + self.l_rate * y_train[i] * X_train[i]
                        self.b = self.b + self.l_rate * y_train[i]
                if flag:
                    is_wrong = True

    model = Perceptron_Model()
    model.fit(X_train, y_train)
    return torch.cat([model.w, torch.unsqueeze(model.b, dim=0)])

print(classification_two(X_train, y_train))
