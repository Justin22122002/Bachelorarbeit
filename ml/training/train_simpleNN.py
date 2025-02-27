import numpy as np
import torch
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset

from ml.cuda.CudaHelper import CudaHelper
from ml.dataset.CICMalMem2022DatasetLoader import CICMalMem2022DatasetLoader
from ml.models.SimpleNN import SimpleNN
from ml.training.TrainerSimpleNN import TrainerSimpleNN


def train_simpleNN() -> None:
    """Function to train the SimpleNN model on the CICMalMem2022 dataset."""

    # Set random seed for reproducibility
    seed: int = 64
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load dataset
    loader: CICMalMem2022DatasetLoader = CICMalMem2022DatasetLoader()
    loader.load_dataset()
    dataset: DataFrame = loader.data_frame

    # Define feature columns
    features: list[str] = [
        'pslist_nproc', 'pslist_nppid', 'pslist_avg_threads', 'pslist_nprocs64bit',
        'pslist_avg_handlers', 'dlllist_ndlls', 'dlllist_avg_dlls_per_proc', 'handles_nhandles',
        'handles_avg_handles_per_proc', 'handles_nport', 'handles_nfile', 'handles_nevent',
        'handles_ndesktop', 'handles_nkey', 'handles_nthread', 'handles_ndirectory',
        'handles_nsemaphore', 'handles_ntimer', 'handles_nsection', 'handles_nmutant',
        'ldrmodules_not_in_load', 'ldrmodules_not_in_init', 'ldrmodules_not_in_mem',
        'ldrmodules_not_in_load_avg', 'ldrmodules_not_in_init_avg', 'ldrmodules_not_in_mem_avg',
        'malfind_ninjections', 'malfind_commitCharge', 'malfind_protection', 'malfind_uniqueInjections',
        'psxview_not_in_pslist', 'psxview_not_in_eprocess_pool', 'psxview_not_in_ethread_pool',
        'psxview_not_in_pspcid_list', 'psxview_not_in_csrss_handles', 'psxview_not_in_session',
        'psxview_not_in_deskthrd', 'psxview_not_in_pslist_false_avg', 'psxview_not_in_eprocess_pool_false_avg',
        'psxview_not_in_ethread_pool_false_avg', 'psxview_not_in_pspcid_list_false_avg',
        'psxview_not_in_csrss_handles_false_avg', 'psxview_not_in_session_false_avg',
        'psxview_not_in_deskthrd_false_avg', 'modules_nmodules', 'svcscan_nservices',
        'svcscan_kernel_drivers', 'svcscan_fs_drivers', 'svcscan_process_services',
        'svcscan_shared_process_services', 'svcscan_interactive_process_services',
        'svcscan_nactive', 'callbacks_ncallbacks', 'callbacks_nanonymous',
        'callbacks_ngeneric',
    ]  # Excludes 'Label', 'SubType', 'Raw_Type'

    # Extract features and labels
    X: np.ndarray = dataset[features].values
    Y: np.ndarray = dataset['Label'].apply(lambda x: 1 if x == 'Malware' else 0).values

    # Split dataset into training and validation sets
    X_train: np.ndarray
    X_val: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=11)

    # Standardize features
    scaler: StandardScaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to PyTorch tensors
    train_dataset: TensorDataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    val_dataset: TensorDataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )

    # Create data loaders
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Get device (CPU or GPU)
    device_helper: CudaHelper = CudaHelper()
    device: torch.device = device_helper.device

    # Initialize model, loss function, and optimizer
    model: SimpleNN = SimpleNN(len(features)).to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=0.01)  # weight_decay default: 1e-2

    # Scheduler
    scheduler: optim.lr_scheduler.StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Train model
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
