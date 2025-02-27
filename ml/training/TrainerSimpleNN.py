import numpy as np
import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn.metrics import classification_report

class TrainerSimpleNN:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: _Loss,
        device: torch.device,
        patience: int = 10,
        scheduler: optim.lr_scheduler = None,
        use_early_stopping: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.scheduler = scheduler
        self.use_early_stopping = use_early_stopping

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> None:
        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 1. Forward pass
                outputs = self.model(inputs)

                # 2. Calculate loss
                loss = self.criterion(outputs, labels.long())

                # 3. Optimizer zero grad
                self.optimizer.zero_grad()

                # 4. Backward pass and optimization / Loss backward (backpropagation)
                loss.backward()

                # 5. Optimizer step (gradient descent)
                self.optimizer.step()
                total_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_accuracy = 100 * correct / total

            # Validation Phase
            val_loss, val_accuracy = self.evaluate(val_loader)

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}% | "
                f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%"
            )

            # Save the best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), "best_model_simpleNN.pth")
                print(f"Best model updated at epoch {epoch + 1} with validation loss {val_loss:.4f}.")

            # Update learning rate using scheduler
            if self.scheduler:
                self.scheduler.step()

            # Early Stopping (if enabled)
            if self.use_early_stopping:
                if val_loss >= self.best_loss:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print("Early stopping triggered!")
                        break
                else:
                    self.patience_counter = 0

    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.long())
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return total_loss / len(loader), accuracy


    def evaluate_model(self, loader: DataLoader):
        """
        Evaluate the model on the given DataLoader and return precision, recall, f1, and accuracy.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate classification report and accuracy
        report = classification_report(all_labels, all_preds, output_dict=True)
        print(f"REPORT:\n{classification_report(all_labels, all_preds)}")
        accuracy = (all_preds == all_labels).mean() * 100

        # Extract the metrics for class 0 and 1
        precision_0 = report.get('0.0', {}).get('precision', 0.0)
        recall_0 = report.get('0.0', {}).get('recall', 0.0)
        f1_0 = report.get('0.0', {}).get('f1-score', 0.0)

        precision_1 = report.get('1.0', {}).get('precision', 0.0)
        recall_1 = report.get('1.0', {}).get('recall', 0.0)
        f1_1 = report.get('1.0', {}).get('f1-score', 0.0)

        return {
            "precision_0": precision_0,
            "recall_0": recall_0,
            "f1_0": f1_0,
            "precision_1": precision_1,
            "recall_1": recall_1,
            "f1_1": f1_1,
            "accuracy": accuracy
        }

