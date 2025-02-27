import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset

from ml.cuda.CudaHelper import CudaHelper
from ml.dataset.CICMalMem2022DatasetLoader import CICMalMem2022DatasetLoader
from ml.models.SimpleNN import SimpleNN
from ml.training.TrainerSimpleNN import TrainerSimpleNN


def train_simpleNN_HW() -> None:
    """Function to train the SimpleNN model using the CICMalMem2022 dataset (HW version)."""

    # Set random seed for reproducibility
    seed: int = 64
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
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
    ]  # Excludes 'Label', 'SubType', 'Raw_Type'

    # Extract features and labels
    X: np.ndarray = dataset[features].values
    Y: np.ndarray = dataset['Label'].apply(lambda x: 1 if x == 'Malware' else 0).values

    # Split dataset into training and validation sets
    X_train: np.ndarray
    X_val: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Standardize features
    scaler: StandardScaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    train_dataset: TensorDataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                 torch.tensor(Y_train, dtype=torch.float32))
    val_dataset: TensorDataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                               torch.tensor(Y_val, dtype=torch.float32))

    # Create data loaders
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=32)

    # Get device (CPU or GPU)
    device_helper: CudaHelper = CudaHelper()
    device: torch.device = device_helper.device

    # Initialize model, loss function, and optimizer
    model: SimpleNN = SimpleNN(len(features)).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    trainer: TrainerSimpleNN = TrainerSimpleNN(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=10,
        use_early_stopping=False
    )

    trainer.train(train_loader, val_loader, epochs=500)

    # Evaluate model
    trainer.evaluate_model(val_loader)

    # print(trainer.model.state_dict())  # Uncomment to print model state dictionary
