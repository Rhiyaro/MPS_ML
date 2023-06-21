from abc import ABC, abstractmethod

import pandas as pd


class Saver(ABC):
    main_folder: str

    def save_results(self, results: dict) -> None:
        for path, data in results.items():
            try:
                if isinstance(data, pd.DataFrame):
                    self.save_dataframe(
                        data=data, path=path, main_folder=self.main_folder
                    )
            except Exception as exc:
                raise Exception(
                    "Something went wrong when saving results: " + str(exc)
                ) from exc

    @abstractmethod
    def save_dataframe(
            self, data: pd.DataFrame, path: str, main_folder: str = None
    ) -> None:
        pass
