from typing import Any

import torch
from torch import nn, optim
from typing_extensions import override

from ml.training.refactored.BaseTrainer import BaseTrainer


class TrainingMulticlassNN(BaseTrainer[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]):

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            loss_fn_multi: nn.Module,
            loss_fn_binary: nn.Module,
            device: torch.device,
            **kwargs: Any
    ) -> None:
        super().__init__(model, optimizer, loss_fn_binary, device, **kwargs)
        self.loss_fn_multi: nn.Module = loss_fn_multi

    @override
    def forward_pass(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(inputs)

    @override
    def calculate_loss(
            self, outputs: tuple[torch.Tensor, torch.Tensor], labels: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        output_multi: torch.Tensor
        output_binary: torch.Tensor
        output_multi, output_binary = outputs

        y_multi: torch.Tensor
        y_binary: torch.Tensor
        y_multi, y_binary = labels

        loss_multi: torch.Tensor = self.loss(output_multi, y_multi)
        loss_binary: torch.Tensor = self.loss_fn_binary(output_binary.squeeze(), y_binary)
        return loss_multi + loss_binary

    @override
    def calculate_accuracy(
            self, outputs: tuple[torch.Tensor, torch.Tensor], labels: tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        output_multi: torch.Tensor
        output_binary: torch.Tensor
        output_multi, output_binary = outputs

        y_multi: torch.Tensor
        y_binary: torch.Tensor
        y_multi, y_binary = labels

        _, preds_multi = torch.max(output_multi, 1)
        accuracy_multi: float = (preds_multi == y_multi).float().mean().item() * 100

        preds_binary: torch.Tensor = (output_binary.squeeze() > 0.5).float()
        accuracy_binary: float = (preds_binary == y_binary).float().mean().item() * 100

        return (accuracy_multi + accuracy_binary) / 2

