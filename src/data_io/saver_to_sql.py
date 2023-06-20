import pandas as pd

from src.data_io.saver import Saver
from src.sql_connection import SQLConnection


class SaverToSQL(Saver):

    def __init__(self, 
                 sql_connection: SQLConnection,
                 main_folder: str = 'output\\results\\') -> None:
        self.sql_connection = sql_connection
        self.main_folder = main_folder

    def save_dataframe(self,
                       data: pd.DataFrame,
                       path: str,
                       main_folder: str = None
                       ) -> None:
        data.to_sql(path,
                    self.sql_connection.engine,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=100)
        self.sql_connection.engine.dispose()
