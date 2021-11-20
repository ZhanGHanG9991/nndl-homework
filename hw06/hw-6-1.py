"""
train_data = torch.randn((6, 6, 8))
train_label = torch.zeros((6, 2))
test_data = torch.randn((3, 6, 8))
epoch_num = 8
ret = torch.tensor([1, 1, 0])
"""
import torch
import torch.nn as nn


def text_classification(train_data, train_label, test_data, epoch_num):
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.rnn = nn.Sequential(
                nn.LSTM(input_size=8, hidden_size=12, batch_first=True),
            )
            self.fnn = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Flatten(1, -1),
                nn.Linear(12 * 6, 2),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            x, _ = self.rnn(x)
            output = self.fnn(x)
            return output

    model = Network()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epoch_num):
        y_pred = model(train_data)
        loss = criterion(y_pred, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_pred = torch.argmax(model(test_data), -1)
    return y_pred
