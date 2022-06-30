from numpy import number
from torch.nn import Module, Sequential, Linear, MSELoss
from torch.utils.data import Dataset
from torch.distributions.uniform import Uniform
from torch.optim import SGD


class LineNetwork(Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = Sequential(
            Linear(1, 1)  # 1 input pra 1 output
        )

    def forward(self, x):
        return self.layers(x)

    def optimizer(self, learning_rate):
        return SGD(self.parameters(), lr=learning_rate)


class AlgebricDataset(Dataset):
    def __init__(self, function, interval, number_samples) -> None:
        X = Uniform(interval[0], interval[1]).sample([number_samples])
        self.data = [(x, function(x)) for x in X]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
