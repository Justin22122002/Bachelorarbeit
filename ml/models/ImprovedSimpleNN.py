import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ImprovedSimpleNN(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(ImprovedSimpleNN, self).__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, 128)  # Larger layer
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(128)  # Batch Normalization
        self.fc2: nn.Linear = nn.Linear(128, 64)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(64)  # Batch Normalization
        self.fc3: nn.Linear = nn.Linear(64, 32)
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(32)  # Batch Normalization
        self.fc4: nn.Linear = nn.Linear(32, 2)  # Output layer for "Malware" and "Benign"
        self.dropout: nn.Dropout = nn.Dropout(0.4)  # Higher dropout value

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = F.relu(self.bn1(self.fc1(x)))  # ReLU with Batch Normalization
        x: Tensor = self.dropout(x)  # Dropout
        x: Tensor = F.relu(self.bn2(self.fc2(x)))  # ReLU with Batch Normalization
        x: Tensor = self.dropout(x)  # Dropout
        x: Tensor = F.relu(self.bn3(self.fc3(x)))  # ReLU with Batch Normalization
        x: Tensor = self.fc4(x)  # Output layer
        return x
