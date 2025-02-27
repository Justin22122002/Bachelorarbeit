import torch
from torch import nn, optim
from collections.abc import Callable
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")  # Generischer Typ für die Outputs
L = TypeVar("L")  # Generischer Typ für die Labels

class BaseTrainer(ABC, Generic[T, L]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: _Loss | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
        patience: int = 10,
        scheduler: optim.lr_scheduler = None,
        use_early_stopping: bool = True,
        best_model_name: str = "best_model.pth"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.patience = patience
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.scheduler = scheduler
        self.use_early_stopping = use_early_stopping
        self.best_model_name = best_model_name

    @abstractmethod
    def forward_pass(self, inputs: torch.Tensor) -> T:
        pass

    @abstractmethod
    def calculate_loss(self, outputs: T, labels: L) -> torch.Tensor:
        pass

    @abstractmethod
    def calculate_accuracy(self, outputs: T, labels: L) -> float:
        pass

    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()

    @staticmethod
    def loss_backward(loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self) -> None:
        self.optimizer.step()

    def early_stopping(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False

    def save_best_model(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(self.model.state_dict(), self.best_model_name)
            print(f"Best model updated with validation loss {val_loss:.4f}.")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100) -> None:
        for epoch in range(epochs):
            self.model.train()
            total_loss: float = 0.0
            total_accuracy: float = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 1. Forward pass
                outputs = self.forward_pass(inputs)

                # 2. Calculate loss
                loss = self.calculate_loss(outputs, labels)

                # 3. Optimizer zero grad
                self.optimizer_zero_grad()

                # 4. Backward pass and optimization / Loss backward (backpropagation)
                self.loss_backward(loss)

                # 5. Optimizer step (gradient descent)
                self.optimizer_step()

                # Calculate training accuracy
                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(outputs, labels)

            avg_loss = total_loss / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)
            val_loss, val_accuracy = self.evaluate(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

            # Save the best model
            self.save_best_model(val_loss)

            # Update learning rate using scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early Stopping (if enabled)
            if self.use_early_stopping and self.early_stopping(val_loss):
                break

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward_pass(inputs)
                loss = self.calculate_loss(outputs, labels)
                total_loss += loss.item()
                total_accuracy += self.calculate_accuracy(outputs, labels)

        return total_loss / len(loader), total_accuracy / len(loader)
