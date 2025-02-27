import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from torch import optim, nn, Tensor
from torch.utils.data import TensorDataset, DataLoader

from ml.cuda.CudaHelper import CudaHelper
from ml.dataset.CICMalMem2022DatasetLoader import CICMalMem2022DatasetLoader
from ml.models.MulticlassNN import MulticlassNN
from ml.training.TrainingMulticlassNN import TrainingMulticlassNN


def train_multiclassNN_with_cross_validation(k: int = 10) -> None:
    """Function to train the MulticlassNN model using k-fold cross-validation."""

    # Set random seed for reproducibility
    seed: int = 64
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    loader: CICMalMem2022DatasetLoader = CICMalMem2022DatasetLoader()
    loader.load_dataset("./datasets/output_file_Run_1_labeled.csv")
    dataset: DataFrame = loader.data_frame

    # Encode categorical labels
    dataset['Label'] = dataset['Label'].map({'Benign': 0, 'Malware': 1})
    dataset['SubType'] = dataset['SubType'].map({'Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3})

    # Extract features and labels
    features: np.ndarray = dataset.drop(columns=['SubType', 'Label']).values
    labels_multiclass: np.ndarray = dataset['SubType'].values
    labels_binary: np.ndarray = dataset['Label'].values

    # Cross-validation setup
    kf: StratifiedKFold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results_multi: list[dict[str, float]] = []
    fold_results_bin: list[dict[str, float]] = []

    # Get device (CPU or GPU)
    device_helper: CudaHelper = CudaHelper()
    device: torch.device = device_helper.device

    for fold, (train_idx, val_idx) in enumerate(kf.split(features, labels_multiclass), 1):
        print(f"Fold {fold}: Training set size = {len(train_idx)}, Validation set size = {len(val_idx)}")

        # Initialize model
        model: MulticlassNN = MulticlassNN(55).to(device)

        # Split data into training and validation sets for the current fold
        X_train: np.ndarray = features[train_idx]
        X_val: np.ndarray = features[val_idx]
        y_train_multiclass: np.ndarray = labels_multiclass[train_idx]
        y_val_multiclass: np.ndarray = labels_multiclass[val_idx]
        y_train_binary: np.ndarray = labels_binary[train_idx]
        y_val_binary: np.ndarray = labels_binary[val_idx]

        # Convert data to PyTorch tensors
        X_train_tensor: Tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor: Tensor = torch.tensor(X_val, dtype=torch.float32)
        y_train_multiclass_tensor: Tensor = torch.tensor(y_train_multiclass, dtype=torch.long)
        y_val_multiclass_tensor: Tensor = torch.tensor(y_val_multiclass, dtype=torch.long)
        y_train_binary_tensor: Tensor = torch.tensor(y_train_binary, dtype=torch.float32)
        y_val_binary_tensor: Tensor = torch.tensor(y_val_binary, dtype=torch.float32)

        # Create data loaders
        train_dataset: TensorDataset = TensorDataset(X_train_tensor, y_train_multiclass_tensor, y_train_binary_tensor)
        val_dataset: TensorDataset = TensorDataset(X_val_tensor, y_val_multiclass_tensor, y_val_binary_tensor)

        train_loader: DataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Define optimizer and learning rate scheduler
        optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        scheduler: optim.lr_scheduler.StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

        # Initialize trainer
        trainer: TrainingMulticlassNN = TrainingMulticlassNN(
            model=model,
            optimizer=optimizer,
            loss_fn_multi=nn.CrossEntropyLoss(),
            loss_fn_binary=nn.BCEWithLogitsLoss(),
            device=device,
            patience=30,
            use_early_stopping=False,
            scheduler=scheduler
        )

        # Train model for this fold
        trainer.train(train_loader, X_val_tensor, y_val_multiclass_tensor, y_val_binary_tensor, epochs=1000)

        # Evaluate model performance on validation set
        result: dict[str, float] = trainer.evaluate_model(
            model, X_val_tensor, y_val_multiclass_tensor, y_val_binary_tensor
        )

        # Collect results for multiclass and binary classification
        fold_results_multi.append(result)
        fold_results_bin.append(result)

        # Print detailed metrics for the current fold
        print(result)

    # Compute average metrics across all folds
    avg_accuracy_multi: float = np.mean([r["accuracy_multi"] for r in fold_results_multi])
    avg_accuracy_bin: float = np.mean([r["accuracy_binary"] for r in fold_results_bin])

    avg_precision_multi: float = np.mean([r["precision_multi"] for r in fold_results_multi])
    avg_recall_multi: float = np.mean([r["recall_multi"] for r in fold_results_multi])
    avg_f1_multi: float = np.mean([r["f1_score_multi"] for r in fold_results_multi])

    avg_precision_bin: float = np.mean([r["precision_binary"] for r in fold_results_bin])
    avg_recall_bin: float = np.mean([r["recall_binary"] for r in fold_results_bin])
    avg_f1_bin: float = np.mean([r["f1_score_binary"] for r in fold_results_bin])

    # Print average performance metrics
    print(f"\nAverage Results over {k} folds:")
    print(f"Multiclass Accuracy: {avg_accuracy_multi:.2f}")
    print(f"Binary Accuracy: {avg_accuracy_bin:.2f}")
    print(f"Multiclass Precision: {avg_precision_multi:.2f}")
    print(f"Multiclass Recall: {avg_recall_multi:.2f}")
    print(f"Multiclass F1-Score: {avg_f1_multi:.2f}")
    print(f"Binary Precision: {avg_precision_bin:.2f}")
    print(f"Binary Recall: {avg_recall_bin:.2f}")
    print(f"Binary F1-Score: {avg_f1_bin:.2f}")
