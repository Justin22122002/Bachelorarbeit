import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset

from ml.cuda.CudaHelper import CudaHelper
from ml.dataset.CICMalMem2022DatasetLoader import CICMalMem2022DatasetLoader
from ml.models.SimpleNN import SimpleNN
from ml.training.TrainerSimpleNN import TrainerSimpleNN


def train_simpleNN_HW_with_cross_validation(k: int = 10) -> None:
    """Function to train the SimpleNN model using k-fold cross-validation."""

    # Set random seed for reproducibility
    seed: int = 64
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and prepare the dataset
    loader: CICMalMem2022DatasetLoader = CICMalMem2022DatasetLoader()
    loader.load_dataset("./datasets/output_file_Run_1_labeled_HW.csv")
    dataset: DataFrame = loader.data_frame

    # Define selected feature columns
    features: list[str] = [
        'malfind_ninjections', 'malfind_commitCharge', 'malfind_protection', 'malfind_uniqueInjections', 'pstree_nTree',
        'pstree_nHandles',
        'pstree_nPID', 'pstree_nPPID', 'pstree_AvgThreads', 'pstree_nWow64', 'pstree_AvgChildren', 'pslist_nproc',
        'pslist_nppid', 'pslist_avg_threads',
        'pslist_nprocs64bit', 'pslist_avg_handlers', 'nConn', 'nDistinctForeignAdd', 'nDistinctForeignPort',
        'nDistinctLocalAddr', 'nDistinctLocalPort',
        'nOwners', 'nDistinctProc', 'nListening', 'Proto_TCPv4', 'Proto_TCPv6', 'Proto_UDPv4', 'Proto_UDPv6',
        'file_total_changes', 'file_added_files',
        'file_modified_files', 'file_deleted_files', 'registry_total_changes', 'registry_added_files',
        'registry_modified_files', 'registry_deleted_files',
    ]

    # Extract features and labels
    X: np.ndarray = dataset[features].values
    Y: np.ndarray = dataset['Label'].apply(lambda x: 1 if x == 'Malware' else 0).values

    # Standardize features
    scaler: StandardScaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Cross-validation setup
    kf: KFold = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results: list[float] = []
    precision_results: list[float] = []
    recall_results: list[float] = []
    f1_results: list[float] = []

    # Get device (CPU or GPU)
    device_helper: CudaHelper = CudaHelper()
    device: torch.device = device_helper.device

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Training fold {fold}/{k}...")

        # Initialize model
        model: SimpleNN = SimpleNN(len(features)).to(device)

        # Split the data into training and validation for the current fold
        X_train: np.ndarray = X[train_idx]
        X_val: np.ndarray = X[val_idx]
        Y_train: np.ndarray = Y[train_idx]
        Y_val: np.ndarray = Y[val_idx]

        # Convert to PyTorch tensors
        train_dataset: TensorDataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                     torch.tensor(Y_train, dtype=torch.float32))
        val_dataset: TensorDataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                   torch.tensor(Y_val, dtype=torch.float32))

        # Create data loaders
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        # Define loss function and optimizer
        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=0.001)

        # Define learning rate scheduler
        scheduler: optim.lr_scheduler.StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

        # Train model for this fold
        trainer: TrainerSimpleNN = TrainerSimpleNN(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            patience=30,
            scheduler=scheduler,
            use_early_stopping=True
        )

        trainer.train(train_loader, val_loader, epochs=1000)

        # Evaluate model performance on validation set
        val_loss: float
        val_accuracy: float
        val_loss, val_accuracy = trainer.evaluate(val_loader)
        fold_results.append(val_accuracy)
        print(f"Fold {fold} | Validation Accuracy: {val_accuracy:.2f}%")

        # Get precision, recall, and F1 score
        metrics: dict[str, float] = trainer.evaluate_model(val_loader)
        precision_results.append(metrics['precision_1'])  # Assuming class '1' represents Malware
        recall_results.append(metrics['recall_1'])
        f1_results.append(metrics['f1_1'])

    # Compute average performance across all folds
    avg_accuracy: float = np.mean(fold_results)
    avg_precision: float = np.mean(precision_results)
    avg_recall: float = np.mean(recall_results)
    avg_f1: float = np.mean(f1_results)

    # Print final results
    print(f"Average Accuracy over {k} folds: {avg_accuracy:.2f}%")
    print(f"Average Precision over {k} folds: {avg_precision:.2f}")
    print(f"Average Recall over {k} folds: {avg_recall:.2f}")
    print(f"Average F1-Score over {k} folds: {avg_f1:.2f}")
