import pandas as pd

from src.ml_model import MLModel
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

    def load_panel_by_model_set(self,
                                panel_model_ids: list[int]
                                ) -> list[int]:
        query = (
            "SELECT panel_id FROM panel"
            "WHERE panel_model_id in "
            f"({str(panel_model_ids)[1:-1]})"
        )

        panel_ids = self.load_something(query)

        return panel_ids["panel_id"].to_list()

    def load_all_failures(self, date_start: str, date_end: str, extra_clauses=None):
        if extra_clauses is None:
            extra_clauses = ["1=1"]

        extra_clauses = ["and " + clause for clause in extra_clauses]
        extra_clauses = " ".join(extra_clauses)

        query = (
                "SELECT panel_id, failure_classification_id, ts_start, ts_end "
                "FROM failure_events "
                "WHERE 1=1 " + extra_clauses
        )

        failures = self.load_something(query)

        failures["ts_end"] = failures["ts_end"].fillna("2100-01-01")

        return failures

    def load_panel_data(self,
                        panel_id: int,
                        channel_ids: list[int],
                        date_start: str,
                        date_end: str) -> pd.DataFrame:

        query = (
            "SELECT ts, channel_id, value "
            "FROM measure "
            f"WHERE ts between '{date_start}' and '{date_end}' "
            f"AND channel_id in ({str(channel_ids)[1:-1]}) "
            f"AND panel_id = {panel_id}"
        )

        data = self.load_something(query)
        data = data.pivot(index="ts", columns="channel_id", values="value")
        data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
        data = data[(data.index >= date_start) & (data.index < date_end)]
        return data

    def load_panel_model(self, panel_id: int) -> int | None:

        query = (
            "SELECT panel_model_id "
            "FROM panel "
            f"WHERE panel_id = {panel_id}"
        )

        panel_model_id = self.load_something(query)

        if panel_model_id.empty:
            return None

        panel_model_id = panel_model_id.iloc[0, 0]

        return panel_model_id

    def load_panel_failures(self, panel_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        query = (
            "SELECT panel_id, alert_classification_id, ts_start, ts_end "
            "FROM failure_event "
            f"WHERE panel_id = {panel_id}"
        )

        panel_failures = self.load_something(query)

        panel_failures = panel_failures.query("ts_start >= @start_date and ts_start <= @end_date")
        panel_failures["ts_end"] = panel_failures["ts_end"].fillna("2100-01-01 00:00:00")

        return panel_failures
