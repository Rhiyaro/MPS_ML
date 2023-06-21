from abc import ABC, abstractmethod

import pandas as pd


class Loader(ABC):
    @abstractmethod
    def load_something(self, path:str) -> pd.DataFrame:
        pass


