import pandas as pd
from pandas import DataFrame


class CICMalMem2022DatasetLoader:
    def __init__(self) -> None:
        """
        Initializes the CICMalMem2022DatasetLoader with a DataFrame placeholder.
        """
        self.data_frame: DataFrame | None = None

    def load_dataset(self, dataset_file_path: str = './datasets/MalwareMemoryDump.csv') -> None:
        """
        Loads the dataset from a CSV file, excluding the 'Raw_Type' column.
        """
        self.data_frame = pd.read_csv(dataset_file_path, usecols=lambda column: column != 'Raw_Type')

    def down_cast(self) -> None:
        """
        Optimizes memory usage of the DataFrame by down casting integer columns to the smallest feasible type
        and converting columns with few unique string values to the 'category' type.
        """
        int8_min: int = -128
        int8_max: int = 127
        int16_min: int = -32768
        int16_max: int = 32767
        int32_min: int = -2147483648
        int32_max: int = 2147483647

        if self.data_frame is not None:
            cols = self.data_frame.columns
            for col in cols:
                if self.data_frame[col].dtype == 'int64':
                    min_val: int = self.data_frame[col].min()
                    max_val: int = self.data_frame[col].max()

                    if min_val == 0 and max_val == 0:
                        self.data_frame = self.data_frame.drop(columns=col)
                        continue

                    if min_val >= int8_min and max_val <= int8_max:
                        self.data_frame[col] = self.data_frame[col].astype('int8')
                    elif min_val >= int16_min and max_val <= int16_max:
                        self.data_frame[col] = self.data_frame[col].astype('int16')
                    elif min_val >= int32_min and max_val <= int32_max:
                        self.data_frame[col] = self.data_frame[col].astype('int32')

                elif self.data_frame[col].dtype == 'O' and self.data_frame[col].nunique() < 5:
                    self.data_frame[col] = self.data_frame[col].astype('category')
