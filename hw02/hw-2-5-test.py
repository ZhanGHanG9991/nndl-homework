"""
X_train = torch.tensor([[0.9, 0.1], [1.0, 1.0], [1.0, 2.2], [0.9, 2.4], [0.8, 1.4],
                        [1.4, 0.4], [1.6, 0.4], [1.7, 0.1], [1.9, 1.0], [0.7, 1.1],
                        [3.6, 6.5], [3.7, 6.0], [4.6, 6.9], [4.4, 6.9], [4.1, 5.5],
                        [4.8, 6.4], [4.1, 6.5], [3.5, 7.5], [3.9, 7.0], [3.4, 5.7],
                        [6.7, 2.1], [6.9, 2.1], [5.8, 2.7], [6.8, 3.2], [6.7, 3.3],
                        [6.7, 2.0], [6.3, 2.5], [6.5, 2.0], [6.2, 3.4], [5.9, 3.0]])
y_train = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

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
                print(y_train)
                print(self.b)
                print(type(self.b))
                is_wrong = False
                while not is_wrong:
                    flag = True
                    for j in range(len(X_train)):
                        s = self.sign(X_train[j], self.w[i], self.b[i])
                        if y_train[j] * s <= 0:
                            print(y_train[j] * s)
                            flag = False
                            print(self.w[i])
                            print(self.b[i])
                            print(self.l_rate * y_train[j])
                            print(self.b[i] + self.l_rate * y_train[j])
                            self.w[i] = self.w[i] + self.l_rate * y_train[j] * X_train[j]
                            self.b[i] = self.b[i] + self.l_rate * y_train[j]
                            print(self.w[i])
                            print(self.b[i])
                    if flag:
                        is_wrong = True

    model = Perceptron_Model()
    model.fit(X_train, y_train)
    print(model.w)
    print(model.b)


classification_mul(X_train, y_train)
# X_train_list = X_train.numpy().tolist()
# y_train_list = y_train.numpy().tolist()
# x1 = [X_train_list[i][0] for i in range(len(X_train_list)) if y_train_list[i] == 0]
# y1 = [X_train_list[i][1] for i in range(len(X_train_list)) if y_train_list[i] == 0]
# x2 = [X_train_list[i][0] for i in range(len(X_train_list)) if y_train_list[i] == 1]
# y2 = [X_train_list[i][1] for i in range(len(X_train_list)) if y_train_list[i] == 1]
# x3 = [X_train_list[i][0] for i in range(len(X_train_list)) if y_train_list[i] == 2]
# y3 = [X_train_list[i][1] for i in range(len(X_train_list)) if y_train_list[i] == 2]
# plt.scatter(x1, y1)
# plt.scatter(x2, y2)
# plt.scatter(x3, y3)
# plt.show()
# print(y_train)
# yy_train = y_train.clone().detach()
# y_train[0] += 1
# print(yy_train)
# print(y_train)

