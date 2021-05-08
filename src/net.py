import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, hidden_size):
        super(Network, self).__init__()
        self.l1 = nn.Linear(2, hidden_size)
        self.l2 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        o = torch.sigmoid(self.l2(h))
        return o


def test():
    net = Network(128)
    x = torch.Tensor([[1.2, 3.3], [2.3, 7.8]])
    y = net(x)
    print(y)


if __name__ == "__main__":
    test()
