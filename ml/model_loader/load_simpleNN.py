import torch
from pandas import DataFrame
import numpy as np

from ml.dataset.AnalysisDataLoader import AnalysisDataLoader
from ml.model_loader.SimpleNNModelLoader import SimpleNNModelLoader

if __name__ == "__main__":
    # Load dataset
    loader: AnalysisDataLoader = AnalysisDataLoader()
    loader.load_dataset(dataset_file_path="../datasets/output_file_Run_1_labeled.csv")

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
    ]

    # Prepare input data for model
    X: np.ndarray = dataset[features].values
    custom_samples: torch.Tensor = torch.tensor(X, dtype=torch.float32)

    # Load deep learning model
    model_path: str = "../saved_models/best_model_simpleNN.pth"
    model_loader: SimpleNNModelLoader = SimpleNNModelLoader(model_path)
    model_loader.load_model()

    # Perform predictions
    predictions: list[str] = model_loader.predict(custom_samples)

    # Counters for classification statistics
    correct_malware: int = 0
    incorrect_malware: int = 0
    correct_benign: int = 0
    incorrect_benign: int = 0

    for i, prediction in enumerate(predictions):
        label: str = dataset.iloc[i]['Label']  # Access row using iloc[i]
        raw_type: str = dataset.iloc[i]['Raw_Type']  # Assuming 'Raw_Type' column exists

        # Compare prediction with the actual label
        if label == 'Malware' and prediction == 'Malware':
            correct_malware += 1
        elif label == 'Malware' and prediction == 'Benign':
            incorrect_malware += 1
        elif label == 'Benign' and prediction == 'Benign':
            correct_benign += 1
        elif label == 'Benign' and prediction == 'Malware':
            incorrect_benign += 1

        print(f"Sample {i+1} (Raw: {raw_type}, predicted as: {prediction}, Label: {label})")

    # Output classification results
    print("\nMalware Classification Results:")
    print(f"Correctly classified Malware: {correct_malware}")
    print(f"Incorrectly classified Malware: {incorrect_malware}")

    print("\nBenign Classification Results:")
    print(f"Correctly classified Benign: {correct_benign}")
    print(f"Incorrectly classified Benign: {incorrect_benign}")
