from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    @abstractmethod
    def load_something(self, path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_panel_by_model_set(self,
                                panel_model_ids: list[int]
                                ) -> list[int]:
        pass

    @abstractmethod
    def load_all_failures(self, date_start: str, date_end: str, extra_clauses=None):
        if extra_clauses is None:
            extra_clauses = ["1=1"]
        raise NotImplemented()

    @abstractmethod
    def load_panel_data(self,
                        panel_id: int,
                        channel_ids: list[int],
                        date_start: str,
                        date_end: str) -> pd.DataFrame:
        raise NotImplemented()

    @abstractmethod
    def load_panel_model(self, panel_id: int) -> int:
        raise NotImplemented()

    @abstractmethod
    def load_panel_failures(self, panel_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplemented()
