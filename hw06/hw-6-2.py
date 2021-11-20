"""
train_data = torch.randn((5, 6, 2))
train_label = torch.zeros((5, 2))
test_data = torch.randn((3, 6, 2))
epoch_num = 5
ret = torch.tensor([0, 1, 1])
"""
import torch
import torch.nn as nn


def dataEnhanceModel(train_data, train_label, test_data, epoch_num):
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.rnn = nn.Sequential(
                nn.LSTM(input_size=2, hidden_size=12, batch_first=True),
            )
            self.fnn = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Flatten(1, -1),
                nn.Linear(6 * 12, 2),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            x, _ = self.rnn(x)
            output = self.fnn(x)
            return output

        def enhance(self, train_x, train_y):
            # 加高斯噪声
            train_x_new1 = train_x + torch.randn_like(train_x)

            # 翻转
            train_x_new2 = torch.flip(train_x, [2])

            # 标准化
            train_x_new3 = torch.normal(train_x)

            train_x_new = torch.cat([train_x, train_x_new1, train_x_new2, train_x_new3])
            train_y_new = torch.cat([train_y, train_y, train_y, train_y])
            return train_x_new, train_y_new

    model = Network()
    train_data, train_label = model.enhance(train_data, train_label)
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.enhance(train_data, train_label)

    for epoch in range(epoch_num):
        y_pred = model(train_data)
        loss = criterion(y_pred, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = torch.argmax(model(test_data), -1)
    return y_pred

