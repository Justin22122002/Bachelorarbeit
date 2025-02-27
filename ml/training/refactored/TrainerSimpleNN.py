from typing import override

import torch

from ml.training.refactored.BaseTrainer import BaseTrainer


class TrainerSimpleNN(BaseTrainer[torch.Tensor, torch.Tensor]):

    @override
    def forward_pass(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    @override
    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(outputs, labels.long())

    @override
    def calculate_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        predicted: torch.Tensor
        _, predicted = torch.max(outputs, 1)

        accuracy: float = (predicted == labels).float().mean().item() * 100
        return accuracy

