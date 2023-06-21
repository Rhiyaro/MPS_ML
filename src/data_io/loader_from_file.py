import pandas as pd

from src.data_io.loader import Loader


class LoaderFromFile(Loader):

    def __init__(self, main_folder: str) -> None:
        self.main_folder = main_folder

        # Could have some pathis in the code

    def load_something(self, path: str) -> pd.DataFrame:
        data = pd.read_csv(path, sep=";", index_col=0)
        return data

    def load_panel_by_model_set(self, panel_model_ids: list[int]) -> list[int]:
        return super().load_panel_by_model_set(panel_model_ids)

    def load_all_failures(self, date_start: str, date_end: str, extra_clauses=None):
        super().load_all_failures(date_start,date_end,extra_clauses)

    def load_panel_data(self,
                        panel_id: int,
                        channel_ids: list[int],
                        date_start: str,
                        date_end: str) -> pd.DataFrame:
        return super().load_panel_data(panel_id,channel_ids,date_start,date_end)

    def load_panel_model(self, panel_id:int) -> int:
        return super().load_panel_model(panel_id)
