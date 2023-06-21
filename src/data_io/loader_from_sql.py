import pandas as pd

from src.tchala import Tchala
from src.data_io.loader import Loader
from src.sql_connection import SQLConnection


class LoaderFromSQL(Loader):
    def __init__(self, sql_connection: SQLConnection) -> None:
        self.sql_connection = sql_connection

    def load_something(
            self, path: str
    ) -> pd.DataFrame:  # pylint: disable=arguments-differ
        try:
            data = pd.read_sql_query(path, self.sql_connection.engine)
            self.sql_connection.engine.dispose()
            return data
        except Exception as exc:
            raise Exception("Something went wrong with the query: " + path) from exc
