import torch
from torch import Tensor
from ml.models.SimpleNN import SimpleNN


class SimpleNNModelLoader:
    def __init__(self, model_file_path: str):
        """Initialize the model loader with the given file path."""
        self.model_file_path: str = model_file_path
        self.model: SimpleNN | None = None

    def load_model(self) -> None:
        """Loads the model from the specified file."""
        self.model = SimpleNN(55)
        self.model.load_state_dict(torch.load(self.model_file_path, weights_only=True))
        self.model.eval()
        print(f"Model successfully loaded from {self.model_file_path}.")

    def predict(self, samples: Tensor) -> list[str]:
        """
        Normalizes the input data and returns the model's predictions.

        :param samples: Tensor containing input data (shape: [n_samples, n_features])
        :return: Predictions of the classes
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Call `load_model()` first.")

        # Model predictions
        with torch.no_grad():
            predictions: Tensor = self.model(samples)
            _, predicted_labels = torch.max(predictions, 1)

        print(predictions)

        # Class mapping
        class_labels: list[str] = ["Benign", "Malware"]
        predicted_classes: list[str] = [class_labels[label] for label in predicted_labels]

        return predicted_classes
