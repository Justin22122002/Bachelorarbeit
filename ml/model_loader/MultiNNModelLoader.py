import torch
from torch import Tensor
from ml.models.MulticlassNN import MulticlassNN

class MultiNNModelLoader:
    def __init__(self, model_file_path: str):
        """Initialize the model loader with the given file path."""
        self.model_file_path: str = model_file_path
        self.model: MulticlassNN | None = None

    def load_model(self) -> None:
        """Loads the model from the specified file."""
        self.model = MulticlassNN(55)
        self.model.load_state_dict(torch.load(self.model_file_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Model successfully loaded from {self.model_file_path}.")

    def predict(self, samples: Tensor) -> tuple[list[str], list[str]]:
        """
        Normalizes the input data and returns the model's predictions.

        :param samples: Tensor containing input data (shape: [n_samples, n_features])
        :return: Predictions for multiclass classification and binary classification
        """
        if self.model is None:
            raise RuntimeError("Model has not been loaded. Call `load_model()` first.")

        with torch.no_grad():
            output_multi: Tensor
            output_binary: Tensor
            output_multi, output_binary = self.model(samples)

            _, predicted_labels_multi = torch.max(output_multi, 1)
            predicted_labels_binary: Tensor = (output_binary > 0.5).long().squeeze()

        class_labels_multi: list[str] = ["Benign", "Spyware", "Trojan", "Ransomware"]
        class_labels_binary: list[str] = ["Benign", "Malware"]

        predicted_classes_multi: list[str] = [class_labels_multi[label] for label in predicted_labels_multi]
        predicted_classes_binary: list[str] = [class_labels_binary[label] for label in predicted_labels_binary]

        return predicted_classes_multi, predicted_classes_binary
