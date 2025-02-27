import torch
from torch import nn, Tensor


class BinarySimpleNN(nn.Module):
    def __init__(self, input_size: int):
        super(BinarySimpleNN, self).__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, 64)
        self.fc2: nn.Linear = nn.Linear(64, 32)
        self.fc3: nn.Linear = nn.Linear(32, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = torch.relu(self.fc1(x))
        x: Tensor = torch.relu(self.fc2(x))
        x: Tensor = self.sigmoid(self.fc3(x))
        return x
