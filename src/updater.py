import torch
import torch.nn as nn
from net import Network


class Updater:
    def __init__(self, net, optimizer):
        self.net = net
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

        # 今回のタスクで使用
        # モデルの入力は0~1の値なので和，差，積はそれぞれ0~2, -1~1, 0~1の値となる
        # ネットワークの出力はsigmoidなので0~1なので和と差の範囲をカバーできない
        # モデルの出力に以下の定数をかけたり足したりすることでモデルの出力と答えのカバー範囲を揃える．
        self.mul_constant = torch.Tensor([2, 2, 1])
        self.sum_constant = torch.Tensor([0, -1, 0])

    def step(self, x, t):
        o = self.net(x)
        y = torch.mul(o, self.mul_constant)
        self.optimizer.zero_grad()
        loss = self._cal_loss(y, t)
        loss.backward()
        self.optimizer.step()
        return loss

    def _cal_loss(self, y, t):
        return self.criterion(y + self.sum_constant, t)


def test():
    import torch.optim as optim
    import numpy as np

    net = Network(128)
    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    updater = Updater(net, opt)

    x = torch.Tensor([[0.4, 0.3], [0.5, 0.9]])
    t = torch.Tensor([[0.7, 0.1, 0.12], [1.4, -0.4, 0.45]])
    mc = np.array([2, 2, 1])
    sc = np.array([0, -1, 0])

    print(net(x).data.numpy() * mc + sc)

    for i in range(10000):
        loss = updater.step(x, t)
        if i % 1000 + 1 == 1000:
            print("iter {} : {}".format(i + 1, loss))

    print(net(x).data.numpy() * mc + sc)


if __name__ == "__main__":
    test()
