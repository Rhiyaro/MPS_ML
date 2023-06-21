import pandas as pd

from src.data_io.loader import Loader


class LoaderFromFile(Loader):
    def __init__(self, main_folder: str) -> None:
        self.main_folder = main_folder

        # Could have some pathis in the code

    def load_something(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path, sep=";", index_col=0)
        return data

