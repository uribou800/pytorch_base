import torch
import numpy as np


class DataLoader:
    def __init__(self, iter_num, batchsize):
        self.iter_num = iter_num
        self.batchsize = batchsize

    def __iter__(self):
        return DataLoaderIterator(self.iter_num, self.batchsize)


class DataLoaderIterator:
    def __init__(self, iter_num, batchsize):
        self.iter_num = iter_num
        self.batchsize = batchsize
        self._pointer = 0

    def __next__(self):
        if self._pointer == self.iter_num:
            raise StopIteration
        data = {}
        input_data = np.random.rand(self.batchsize, 2)
        output_data = np.array([input_data[:, 0] + input_data[:, 1],
                                input_data[:, 0] - input_data[:, 1],
                                input_data[:, 0] * input_data[:, 1]]).T
        data["input"] = torch.Tensor(input_data)
        data["output"] = torch.Tensor(output_data)
        self._pointer += 1
        return data


def main():
    data_loader = DataLoader(3, 4)
    for data in data_loader:
        print(data)


if __name__ == "__main__":
    main()
