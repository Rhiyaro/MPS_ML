from src.data_io.loader import Loader
from src.data_io.loader_from_file import LoaderFromFile
from src.data_io.loader_from_sql import LoaderFromSQL
from src.data_io.saver import Saver
from src.data_io.saver_to_file import SaverToFile
from src.data_io.saver_to_sql import SaverToSQL
from src.sql_connection import SQLConnection


class DataIO:  # pylint: disable=too-few-public-methods
    def __init__(self, loader: Loader, saver: Saver) -> None:
        self.loader = loader
        self.saver = saver

    @classmethod
    def create(
        cls,
        sql_connection: SQLConnection = None,
        load_from_sql: bool = False,
        save_to_sql: bool = False,
        main_folder: str = "output\\test_runs",
    ):

        loader = LoaderFromFile(main_folder=main_folder)
        if load_from_sql:
            loader = LoaderFromSQL(sql_connection)

        saver = SaverToFile(main_folder=main_folder)
        if save_to_sql:
            saver = SaverToSQL(sql_connection, main_folder=main_folder)

        data_io = DataIO(loader, saver)

        return data_io
