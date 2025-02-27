import torch
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter


# Train model
# --> To train our model, we're going to need to build a training loop with the following steps:
# - Forward pass
# - Calculate the loss
# - Optimizer zero grad
# - Loss backward (backpropagation)
# - Optimizer step (gradient descent)
# - define Early Stopping


class TrainingMulticlassNN:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn_multi: _Loss,
        loss_fn_binary: _Loss,
        device: torch.device,
        patience: int = 30,
        scheduler: optim.lr_scheduler = None,
        use_early_stopping: bool = True
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn_multi = loss_fn_multi
        self.loss_fn_binary = loss_fn_binary
        self.device = device
        self.patience = patience
        self.best_loss: float = float('inf')
        self.patience_counter: int = 0
        self.writer = SummaryWriter()
        self.scheduler = scheduler
        self.use_early_stopping = use_early_stopping

    def forward_pass(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        output_multi, output_binary = self.model(x)
        return output_multi, output_binary

    def calculate_loss(
        self,
        output_multi: torch.Tensor,
        output_binary: torch.Tensor,
        y_multi: torch.Tensor,
        y_binary: torch.Tensor
    ) -> torch.Tensor:
        y_multi, y_binary = y_multi.to(self.device), y_binary.to(self.device)
        loss_multi: torch.Tensor = self.loss_fn_multi(output_multi, y_multi)
        loss_binary: torch.Tensor = self.loss_fn_binary(output_binary.squeeze(), y_binary)
        loss: torch.Tensor = loss_multi + loss_binary
        return loss

    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def loss_backward(
        self,
        loss: torch.Tensor
    ) -> None:
        loss.backward()

    def optimizer_step(self) -> None:
        self.optimizer.step()

    def early_stopping(
        self,
        avg_loss: float
    ) -> bool:
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False

    def train(
        self,
        train_loader: DataLoader,
        x_test: torch.Tensor,
        y_test_multi: torch.Tensor,
        y_test_binary: torch.Tensor,
        epochs: int = 1000,
    ) -> None:

        for epoch in range(epochs):
            self.model.train()  # Set model to train mode
            total_loss = 0
            total_accuracy_multi = 0
            total_accuracy_binary = 0

            for batch in train_loader:
                x_batch, y_batch_multi, y_batch_binary = batch
                x_batch, y_batch_multi, y_batch_binary = x_batch.to(self.device), y_batch_multi.to(
                    self.device), y_batch_binary.to(self.device)

                # 1. Forward pass
                output_multi, output_binary = self.forward_pass(x_batch)

                # 2. Calculate loss
                loss = self.calculate_loss(output_multi, output_binary, y_batch_multi, y_batch_binary)

                # 3. Optimizer zero grad
                self.optimizer.zero_grad()

                # 4. Backward pass and optimization / Loss backward (backpropagation)
                loss.backward()

                # 5. Optimizer step (gradient descent)
                self.optimizer.step()

                total_loss += loss.item()

                # Calculate accuracy
                _, preds_multi = torch.max(output_multi, 1)
                accuracy_multi = (preds_multi == y_batch_multi).float().mean()

                preds_binary = (output_binary.squeeze() > 0.5).float()
                accuracy_binary = (preds_binary == y_batch_binary).float().mean()

                total_accuracy_multi += accuracy_multi.item()
                total_accuracy_binary += accuracy_binary.item()

            # Calculate average metrics for training
            avg_loss = total_loss / len(train_loader)
            avg_accuracy_multi = total_accuracy_multi / len(train_loader)
            avg_accuracy_binary = total_accuracy_binary / len(train_loader)

            # Evaluate on test data
            test_loss, test_accuracy_multi, test_accuracy_binary = self.evaluate(x_test, y_test_multi, y_test_binary)

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Accuracy/train_multi', avg_accuracy_multi, epoch)
            self.writer.add_scalar('Accuracy/train_binary', avg_accuracy_binary, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/test_multi', test_accuracy_multi, epoch)
            self.writer.add_scalar('Accuracy/test_binary', test_accuracy_binary, epoch)

            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.4f}, Train Multiclass Accuracy: {avg_accuracy_multi:.4f}, '
                f'Train Binary Accuracy: {avg_accuracy_binary:.4f}, Test Loss: {test_loss:.4f}, '
                f'Test Multiclass Accuracy: {test_accuracy_multi:.4f}, Test Binary Accuracy: {test_accuracy_binary:.4f}')

            # Save the best model
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                torch.save(self.model.state_dict(), "best_model_multiclassNN.pth")
                print(f"Best model updated at epoch {epoch + 1} with validation loss {test_loss:.4f}.")

            if self.scheduler:
                self.scheduler.step()

            # 6. Early Stopping
            if self.use_early_stopping:
                trigger = self.early_stopping(avg_loss)
                if trigger: break

        self.writer.close()

    def evaluate(
        self,
        x_test: torch.Tensor,
        y_test_multi: torch.Tensor,
        y_test_binary: torch.Tensor
    ) -> Tuple[float, float, float]:
        self.model.eval()
        with torch.no_grad():
            x_test, y_test_multi, y_test_binary = x_test.to(self.device), y_test_multi.to(self.device), y_test_binary.to(self.device)
            output_multi, output_binary = self.forward_pass(x_test)
            loss = self.calculate_loss(output_multi, output_binary, y_test_multi, y_test_binary)

            _, preds_multi = torch.max(output_multi, 1)
            accuracy_multi: float = (preds_multi == y_test_multi).float().mean().item()

            predicts_binary: torch.Tensor = (output_binary.squeeze() > 0.5).float()
            accuracy_binary: float = (predicts_binary == y_test_binary).float().mean().item()

        return loss.item(), accuracy_multi, accuracy_binary

    def evaluate_model(self, model, x_test, y_test_multi, y_test_binary) -> dict[str, dict | float]:
        """
        Evaluate the model on the given data and print classification reports for both
        multiclass and binary classification.
        """
        model.eval()

        with torch.no_grad():
            # Move inputs and labels to the device
            x_test, y_test_multi, y_test_binary = (
                x_test.to(self.device),
                y_test_multi.to(self.device),
                y_test_binary.to(self.device),
            )

            # Get the model's outputs
            output_multi, output_binary = model(x_test)

            # Multiclass Prediction
            _, preds_multi = torch.max(output_multi, 1)

            # Binary Prediction
            preds_binary = (output_binary.squeeze() > 0.5).float()

            # Calculate accuracy for both multiclass and binary
            accuracy_multi = (preds_multi == y_test_multi).float().mean().item()
            accuracy_binary = (preds_binary == y_test_binary).float().mean().item()

            # Precision, Recall, and F1-Score for Multiclass
            precision_multi = precision_score(y_test_multi.cpu(), preds_multi.cpu(), average='weighted', zero_division=1)
            recall_multi = recall_score(y_test_multi.cpu(), preds_multi.cpu(), average='weighted', zero_division=1)
            f1_multi = f1_score(y_test_multi.cpu(), preds_multi.cpu(), average='weighted', zero_division=1)

            # Precision, Recall, and F1-Score for Binary
            precision_binary = precision_score(y_test_binary.cpu(), preds_binary.cpu(), average='weighted')
            recall_binary = recall_score(y_test_binary.cpu(), preds_binary.cpu(), average='weighted')
            f1_binary = f1_score(y_test_binary.cpu(), preds_binary.cpu(), average='weighted')

            # Print detailed classification reports
            print(f"Multiclass Classification Report:\n{classification_report(y_test_multi.cpu(), preds_multi.cpu(), zero_division=1)}")
            print(f"Binary Classification Report:\n{classification_report(y_test_binary.cpu(), preds_binary.cpu(), zero_division=1)}")

            # Print accuracy for both multiclass and binary
            print(f"Multiclass Accuracy: {accuracy_multi:.4f}")
            print(f"Binary Accuracy: {accuracy_binary:.4f}")

            return {
                "multiclass_report": classification_report(y_test_multi.cpu(), preds_multi.cpu(), output_dict=True, zero_division=1),
                "binary_report": classification_report(y_test_binary.cpu(), preds_binary.cpu(), output_dict=True, zero_division=1),
                "accuracy_multi": accuracy_multi,
                "accuracy_binary": accuracy_binary,
                "precision_multi": precision_multi,
                "recall_multi": recall_multi,
                "f1_score_multi": f1_multi,
                "precision_binary": precision_binary,
                "recall_binary": recall_binary,
                "f1_score_binary": f1_binary
            }

