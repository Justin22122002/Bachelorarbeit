import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import optim, nn, Tensor
from torch.utils.data import TensorDataset, DataLoader

from ml.cuda.CudaHelper import CudaHelper
from ml.dataset.CICMalMem2022DatasetLoader import CICMalMem2022DatasetLoader
from ml.models.MulticlassNN import MulticlassNN
from ml.training.TrainingMulticlassNN import TrainingMulticlassNN


def train_multiclassNN() -> None:
    """Function to train the MulticlassNN model on the CICMalMem2022 dataset."""

    # Load dataset
    loader: CICMalMem2022DatasetLoader = CICMalMem2022DatasetLoader()
    loader.load_dataset()
    dataset: DataFrame = loader.data_frame
    print(dataset.columns)

    # Encode categorical labels
    dataset['Label'] = dataset['Label'].map({'Benign': 0, 'Malware': 1})
    dataset['SubType'] = dataset['SubType'].map({'Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3})

    # Split dataset into training and test sets
    x_train: DataFrame
    x_test: DataFrame
    x_train, x_test = train_test_split(dataset, test_size=0.17, stratify=dataset[['SubType', 'Label']], random_state=42)

    # Extract features and labels
    x_train_feats: Tensor = torch.tensor(x_train.drop(columns=['SubType', 'Label']).values, dtype=torch.float32)
    x_test_feats: Tensor = torch.tensor(x_test.drop(columns=['SubType', 'Label']).values, dtype=torch.float32)

    y_train: Tensor = torch.tensor(x_train[['SubType', 'Label']].values, dtype=torch.float32)
    y_test: Tensor = torch.tensor(x_test[['SubType', 'Label']].values, dtype=torch.float32)

    # Convert target variables
    y_train_multiclass: Tensor = y_train[:, 0].long()
    y_test_multiclass: Tensor = y_test[:, 0].long()

    y_train_binary: Tensor = y_train[:, 1].float()
    y_test_binary: Tensor = y_test[:, 1].float()

    # Get device (CPU or GPU)
    device_helper: CudaHelper = CudaHelper()
    device: torch.device = device_helper.device

    # Initialize model and move to device
    model: MulticlassNN = MulticlassNN(55).to(device)

    # Initialize trainer
    trainer: TrainingMulticlassNN = TrainingMulticlassNN(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.001, weight_decay=0),
        loss_fn_multi=nn.CrossEntropyLoss(),
        loss_fn_binary=nn.BCEWithLogitsLoss(),
        device=device,
        patience=30
    )

    # Create DataLoader for training
    train_data: TensorDataset = TensorDataset(x_train_feats, y_train_multiclass, y_train_binary)
    train_loader: DataLoader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Train the model
    trainer.train(train_loader, x_test_feats, y_test_multiclass, y_test_binary, epochs=100)

    # Evaluate the model after training
    trainer.evaluate_model(model, x_test_feats, y_test_multiclass, y_test_binary)
