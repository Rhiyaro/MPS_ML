from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    @abstractmethod
    def load_something(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_turbine_data(
        self,
        turbine_reg_id: int,
        channel_reg_ids: list[int],
        date_start: str,
        date_end: str,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_turbine_model(self, turbine_reg_id: int) -> int:
        pass

    @abstractmethod
    def load_turbine_pnom(self, turbine_reg_id: int) -> int:
        pass

    @abstractmethod
    def load_turbine_status_ok(self, turbine_reg_id: int) -> int | None:
        pass

    @abstractmethod
    def load_table_failures(
        self, turbine_reg_id: int, date_start: str, date_end: str
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_all_failures(self, date_start: str, date_end: str, extra_clauses:str = "") -> pd.DataFrame:
        pass

    @abstractmethod
    def load_downtimes(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_turbines_by_model(
        self,
        turbine_model_reg_ids: list[int] | int,
    ) -> list[int]:
        pass

    @abstractmethod
    def get_channel_subsystem(self, channel_reg_id: int) -> int:
        pass

    @abstractmethod
    def get_generated_alerts(self, start_date: str, end_date: str, include_aggregated: bool = False) -> pd.DataFrame:
        pass
