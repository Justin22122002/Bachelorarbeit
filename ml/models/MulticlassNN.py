import torch
import torch.nn as nn
from torch import Tensor


class MulticlassNN(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(MulticlassNN, self).__init__()

        self.layer1: nn.Linear = nn.Linear(input_size, 512)
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(512)
        self.act1: nn.ReLU = nn.ReLU()

        self.layer2: nn.Linear = nn.Linear(512, 256)
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(256)
        self.act2: nn.ReLU = nn.ReLU()

        self.layer3: nn.Linear = nn.Linear(256, 128)
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.act3: nn.ReLU = nn.ReLU()

        self.layer4: nn.Linear = nn.Linear(128, 64)
        self.bn4: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.act4: nn.ReLU = nn.ReLU()

        self.layer5: nn.Linear = nn.Linear(64, 32)
        self.bn5: nn.BatchNorm1d = nn.BatchNorm1d(32)
        self.act5: nn.ReLU = nn.ReLU()

        self.layer6: nn.Linear = nn.Linear(32, 16)
        self.bn6: nn.BatchNorm1d = nn.BatchNorm1d(16)
        self.act6: nn.ReLU = nn.ReLU()

        self.output_multi: nn.Linear = nn.Linear(16, 4)  # Multiclass output
        self.output_binary: nn.Linear = nn.Linear(16, 1)  # Binary output

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x: Tensor = self.act1(self.bn1(self.layer1(x)))
        x: Tensor = self.act2(self.bn2(self.layer2(x)))
        x: Tensor = self.act3(self.bn3(self.layer3(x)))
        x: Tensor = self.act4(self.bn4(self.layer4(x)))
        x: Tensor = self.act5(self.bn5(self.layer5(x)))
        x: Tensor = self.act6(self.bn6(self.layer6(x)))

        output_multi: Tensor = self.output_multi(x)  # Multiclass classification output
        output_binary: Tensor = torch.sigmoid(self.output_binary(x))  # Sigmoid for binary classification

        return output_multi, output_binary
