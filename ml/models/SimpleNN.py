import torch
import torch.nn as nn
from torch import Tensor


class SimpleNN(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, 64)  # First fully connected layer
        self.fc2: nn.Linear = nn.Linear(64, 32)  # Second fully connected layer
        self.fc3: nn.Linear = nn.Linear(32, 2)  # Output layer with 2 neurons for 'Malware' and 'Benign'

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = torch.relu(self.fc1(x))  # Apply ReLU activation after first layer
        x: Tensor = torch.relu(self.fc2(x))  # Apply ReLU activation after second layer
        x: Tensor = self.fc3(x)  # Output layer (logits for classification)
        return x
