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


def train_simpleNN_with_cross_validation(k: int = 10) -> None:
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
    loader.load_dataset("./datasets/output_file_Run_1_labeled.csv")
    dataset: DataFrame = loader.data_frame

    # Define selected feature columns
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
            patience=15,
            scheduler=scheduler,
            use_early_stopping=False
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
