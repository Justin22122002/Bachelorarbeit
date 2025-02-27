import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ml.dataset.AnalysisDataLoader import AnalysisDataLoader

if __name__ == '__main__':
    # Directory containing saved models
    model_dir: str = "../saved_models_ml/saved_models"

    # List of model filenames (models must be saved beforehand)
    model_names: list[str] = [
        "RandomForestClassifier_MalwareMemoryDump.pkl",
        "AdaBoostClassifier_MalwareMemoryDump.pkl",
        "LGBMClassifier_MalwareMemoryDump.pkl",
        "GradientBoostingClassifier_MalwareMemoryDump.pkl",
        "StackingClassifier_MalwareMemoryDump.pkl",
        "GaussianNB_MalwareMemoryDump.pkl",
        "SVC_MalwareMemoryDump.pkl"
    ]

    # Load dataset
    loader: AnalysisDataLoader = AnalysisDataLoader()
    loader.load_dataset(dataset_file_path="../datasets/output_file_Run_1_labeled.csv")
    dataset: pd.DataFrame = loader.data_frame

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
    ]

    # Define labels
    X: pd.DataFrame = dataset[features]
    y: pd.DataFrame = dataset[["SubType", "Label"]].copy()  # Multi-label classification: SubType (malware type) and Label (Benign/Malware)

    # Normalize feature values
    scaler: MinMaxScaler = MinMaxScaler()
    X_scaled: pd.DataFrame = scaler.fit_transform(X)

    # Load and evaluate models
    loaded_models: list[tuple[str, object]] = []
    for i, model_name in enumerate(model_names):
        model_path: str = os.path.join(model_dir, model_name)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            loaded_models.append((model_name, model))
            print(f"Loaded: {model_name}")

            y_pred = model.predict(X_scaled)

            # Counters for classification statistics
            correct_malware: int = 0
            incorrect_malware: int = 0
            correct_benign: int = 0
            incorrect_benign: int = 0

            for j, prediction in enumerate(y_pred):
                label: str = dataset.iloc[j]['Label']
                raw_type: str = dataset.iloc[j]['Raw_Type']

                pred_label: str = "Malware" if prediction[1] == 1 else "Benign"  # Label classification
                subtype_label: int = prediction[0]  # SubType classification

                if label == "Malware" and pred_label == "Malware":
                    correct_malware += 1
                elif label == "Malware" and pred_label == "Benign":
                    incorrect_malware += 1
                elif label == "Benign" and pred_label == "Benign":
                    correct_benign += 1
                elif label == "Benign" and pred_label == "Malware":
                    incorrect_benign += 1

                # print(f"Sample {j+1} (Raw: {raw_type}, SubType: {subtype_label}, Predicted as: {pred_label}, Label: {label})")

            # Results for Malware & Benign classifications
            print("\nMalware classification results:")
            print(f"Correctly classified Malware: {correct_malware}")
            print(f"Incorrectly classified Malware: {incorrect_malware}")

            print("\nBenign classification results:")
            print(f"Correctly classified Benign: {correct_benign}")
            print(f"Incorrectly classified Benign: {incorrect_benign}")

        else:
            print(f"Model not found: {model_name}")
