import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

file_path: str = "../datasets/output_file_Run_1_labeled_HW.csv"
dataset: pd.DataFrame = pd.read_csv(file_path)

X: pd.DataFrame = dataset.drop(columns=["Label", "SubType", "Raw_Type"])
y: pd.DataFrame = dataset[["Label", "SubType", "Raw_Type"]]

X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.DataFrame
y_test: pd.DataFrame

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["Label"]
)

noise_level: float = 0.01
augmented_data: pd.DataFrame = X_train + np.random.normal(0, noise_level, X_train.shape)

synthetic_data: pd.DataFrame = X_train.sample(frac=5.0, replace=True).reset_index(drop=True)
synthetic_labels: pd.DataFrame = y_train.sample(frac=5.0, replace=True).reset_index(drop=True)

X_combined: pd.DataFrame = pd.concat([X_train, augmented_data, synthetic_data], ignore_index=True)
y_combined: pd.DataFrame = pd.concat([y_train, y_train, synthetic_labels], ignore_index=True)

augmented_dataset: pd.DataFrame = X_combined.copy()
augmented_dataset["Label"] = y_combined["Label"]
augmented_dataset["SubType"] = y_combined["SubType"]
augmented_dataset["Raw_Type"] = y_combined["Raw_Type"]

output_path: str = "../datasets/augmented_output_file_Run_1_labeled_HW.csv"
augmented_dataset.to_csv(output_path, index=False)

print("Original-Datensatzgröße:", dataset.shape)
print("Erweiterter Datensatz gespeichert unter:", output_path)
print("Neue Größe des Datensatzes:", augmented_dataset.shape)
