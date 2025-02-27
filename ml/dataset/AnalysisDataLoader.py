import pandas as pd
from pandas import DataFrame


class AnalysisDataLoader:
    def __init__(self) -> None:
        """
        Initializes the created Dataset form the malwareAnalysis with a DataFrame placeholder.
        """
        self.data_frame: DataFrame | None = None

    def load_dataset(self, dataset_file_path: str = '../datasets/output_file_Run_1_labeled.csv') -> None:
        """
        Loads the dataset from a CSV file, excluding the 'Raw_Type' column.
        """
        self.data_frame = pd.read_csv(dataset_file_path)