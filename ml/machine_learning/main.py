import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from missforest.missforest import MissForest

# Load dataset
df: pd.DataFrame = pd.read_csv('../datasets/output_file_Run_1_labeled.csv', usecols=lambda column: column != 'Raw_Type')
df['Label'] = df['Label'].map({'Benign': 0, 'Malware': 1})
df['SubType'] = df['SubType'].map({'Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3})

# Splitting the dataset
test_size: float = 0.2
x_train: pd.DataFrame
x_test: pd.DataFrame

x_train, x_test = train_test_split(df, test_size=test_size, stratify=df[['SubType', 'Label']], random_state=42)

x_train_feats: pd.DataFrame = x_train.drop(columns=['SubType', 'Label'])
x_test_feats: pd.DataFrame = x_test.drop(columns=['SubType', 'Label'])
y_train: pd.DataFrame = x_train[['SubType', 'Label']]
y_test: pd.DataFrame = x_test[['SubType', 'Label']]

# Imputation of missing values
mf: MissForest = MissForest()
x_train_feats_imputed: pd.DataFrame = (
    mf.fit_transform(x_train_feats) if x_train_feats.isnull().sum().sum() > 0 else x_train_feats.copy()
)

# Normalization
scaler: MinMaxScaler = MinMaxScaler()
x_train_feats_imputed_sc: pd.DataFrame = pd.DataFrame(
    scaler.fit_transform(x_train_feats_imputed),
    columns=x_train_feats.columns,
    index=y_train.index
)

x_test_feats_sc: pd.DataFrame = pd.DataFrame(
    scaler.transform(x_test_feats),
    columns=x_test_feats.columns,
    index=y_test.index
)

# Define models
models: list = [
    RandomForestClassifier(random_state=42),
    MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                             n_estimators=30, learning_rate=.01,
                                             random_state=42), n_jobs=-1),
    MultiOutputClassifier(LGBMClassifier(random_state=42), n_jobs=-1),
    MultiOutputClassifier(GradientBoostingClassifier(random_state=42), n_jobs=-1),
    MultiOutputClassifier(StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
            ('rf', RandomForestClassifier(max_depth=10, random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ],
        final_estimator=RandomForestClassifier(random_state=42),
        cv=3, n_jobs=-1, verbose=1), n_jobs=-1),
    MultiOutputClassifier(GaussianNB(), n_jobs=-1),
    MultiOutputClassifier(SVC(kernel='rbf', probability=True, random_state=42), n_jobs=-1)
]

# Train models
for i, model in enumerate(models):
    print(f'Training model {i + 1}/{len(models)}: {model.__class__.__name__}')
    model.fit(x_train_feats_imputed_sc, y_train)

# Cross-validation setup
cv_folds: int = 5
cv_strategy: StratifiedKFold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

# Evaluate models
for i, model in enumerate(models):
    y_test_preds: np.ndarray = model.predict(x_test_feats_sc)
    print(f'\n Evaluating model {i + 1}/{len(models)}: {model.__class__.__name__}')

    print('SubType Classification Report:\n',
          classification_report(y_test.iloc[:, 0], y_test_preds[:, 0], zero_division=0))
    print('Label Classification Report:\n',
          classification_report(y_test.iloc[:, 1], y_test_preds[:, 1], zero_division=0))

    # Perform cross-validation for each model
    if isinstance(model, MultiOutputClassifier):
        for label_index, label_name in enumerate(["SubType", "Label"]):
            scores = cross_validate(model.estimators_[label_index], x_train_feats_imputed_sc,
                                    y_train.iloc[:, label_index], cv=cv_strategy,
                                    scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
                                    return_estimator=True, n_jobs=-1)

            print(f'\n Cross-Validation results for {label_name}:')
            print(f'Average Accuracy ({cv_folds}-Fold CV): {np.mean(scores["test_accuracy"]):.4f}')
            print(f'Average Precision ({cv_folds}-Fold CV): {np.mean(scores["test_precision_macro"]):.4f}')
            print(f'Average Recall ({cv_folds}-Fold CV): {np.mean(scores["test_recall_macro"]):.4f}')
            print(f'Average F1-Score ({cv_folds}-Fold CV): {np.mean(scores["test_f1_macro"]):.4f}')

    else:
        scores = cross_validate(model, x_train_feats_imputed_sc, y_train.iloc[:, 1], cv=cv_strategy,
                                scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
                                return_estimator=True, n_jobs=-1)

        print(f'Average Accuracy ({cv_folds}-Fold CV): {np.mean(scores["test_accuracy"]):.4f}')
        print(f'Average Precision ({cv_folds}-Fold CV): {np.mean(scores["test_precision_macro"]):.4f}')
        print(f'Average Recall ({cv_folds}-Fold CV): {np.mean(scores["test_recall_macro"]):.4f}')
        print(f'Average F1-Score ({cv_folds}-Fold CV): {np.mean(scores["test_f1_macro"]):.4f}')

# Save models
model_dir: str = "saved_models"
os.makedirs(model_dir, exist_ok=True)

for i, model in enumerate(models):
    base_model_name: str = (
        model.estimator.__class__.__name__ if isinstance(model, MultiOutputClassifier) else model.__class__.__name__
    )

    model_name: str = f"{base_model_name}_output_file_Run_1_labeled.pkl"
    model_path: str = os.path.join(model_dir, model_name)

    # Save model
    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")
