import os

import pandas as pd

from numpy import nan

from src.data_io.saver import Saver


class SaverToFile(Saver):

    def __init__(self,
                 main_folder: str = 'output\\test_runs') -> None:
        self.main_folder = main_folder

    def save_dataframe(self,
                       data: pd.DataFrame,
                       path: str,
                       main_folder: str = None
                       ) -> None:
        
        if main_folder is None:
            main_folder = self.main_folder
        
        complete_path = f'{main_folder}\\{path}.csv'

        header = True
        mode = 'w'

        if os.path.isfile(complete_path):
            header = False
            mode = 'a'

            data = self.format_data(data, complete_path)

        data.to_csv(complete_path,
                    sep=';',
                    mode=mode,
                    header=header,
                    index=False)
        
    def format_data(self, data:pd.DataFrame, path:str):

        target_file = pd.read_csv(path, sep=";")

        target_cols = list(target_file.columns)

        for col in target_file.columns:
            if col not in data.columns:
                data[col] = nan

        data = data[target_cols]

        return data

