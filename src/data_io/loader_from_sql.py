import pandas as pd

from src.tchala import Tchala
from src.data_io.loader import Loader
from src.sql_connection import SQLConnection


class LoaderFromSQL(Loader):
    def __init__(self, sql_connection: SQLConnection) -> None:
        self.sql_connection = sql_connection

    def load_something(
        self, query: str
    ) -> pd.DataFrame:  # pylint: disable=arguments-differ
        try:
            data = pd.read_sql_query(query, self.sql_connection.engine)
            self.sql_connection.engine.dispose()
            return data
        except Exception as exc:
            raise Exception("Something went wrong with the query: " + query) from exc

    def load_turbine_data(
        self,
        turbine_reg_id: int,
        channel_reg_ids: list[int],
        date_start: str,
        date_end: str,
    ) -> pd.DataFrame:
        try:
            query = (
                "select TS, CHANNEL_REG_ID, VALUE "
                "from turbine with(nolock) "
                f"where TS between '{date_start}' and '{date_end}' "
                f"and CHANNEL_REG_ID in ({str(channel_reg_ids)[1:-1]}) "
                f"and TURBINE_REG_ID={turbine_reg_id}"
            )
            data = self.load_something(query)
            data = data.pivot(index="TS", columns="CHANNEL_REG_ID", values="VALUE")
            data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
            data = data[(data.index >= date_start) & (data.index < date_end)]
            return data
        except Exception as exc:
            raise Exception(
                f"Something went wrong loading turbine data from SQL: {exc}"
            ) from exc

    def load_turbine_model(self, turbine_reg_id: int) -> int:
        query = (
            "select TURBINE_MODEL_REG_ID "
            "from turbine_reg "
            f"where TURBINE_REG_ID = {turbine_reg_id}"
        )

        turbine_model_reg_id = self.load_something(query)
        if turbine_model_reg_id.empty:
            turbine_model_reg_id = None
        else:
            turbine_model_reg_id = turbine_model_reg_id.iloc[0, 0]

        return int(turbine_model_reg_id)

    def load_turbine_pnom(self, turbine_reg_id: int) -> int:
        try:
            query = (
                "select TURBINE_MODEL_REG_PWR from turbine_reg "
                "inner join turbine_model_reg "
                "on turbine_reg.turbine_model_reg_id "
                "= turbine_model_reg.turbine_model_reg_id "
                f"where turbine_reg_id={turbine_reg_id}"
            )
            pnom = self.load_something(query)
            if pnom.empty:
                pnom = None
            else:
                pnom = pnom.iloc[0, 0]
            return pnom
        except Exception as exc:
            raise Exception(
                "Something went wrong loading the value of " "nominal power from SQL"
            ) from exc

    def load_turbine_status_ok(self, turbine_reg_id: int) -> int | None:
        try:
            query = (
                "with t1 as "
                "(select * from turbine_model_status "
                "where turbine_status_reg_id=14) "
                "select turbine_status_code from turbine_reg "
                "inner join turbine_model_reg on "
                "turbine_reg.turbine_model_reg_id = "
                "turbine_model_reg.turbine_model_reg_id "
                "inner join t1 on turbine_reg.turbine_model_reg_id "
                "= t1.turbine_model_reg_id "
                f"where turbine_reg_id={turbine_reg_id}"
            )
            status_ok = self.load_something(query)
            if status_ok.empty:
                status_ok = None
            else:
                status_ok = status_ok.iloc[0, 0]
            return status_ok
        except Exception as exc:
            raise Exception(
                "Something went wrong loading the value of " "status ok from SQL"
            ) from exc

    def load_table_failures(
        self, turbine_reg_id: int, date_start: str, date_end: str
    ) -> pd.DataFrame:
        try:
            query = (
                "select TURBINE_REG_ID, DEFECT_IDENTIFICATION_DATE, "
                "ACTION_DATE, FAULT_DATE, ALARM_SUBSYSTEM_REG_ID, ACTION "
                "from major_component_failure "
                "where IS_MANUFACTURING_ISSUE = 0"
            )
            failures = self.load_something(query)
            if not failures.empty:
                failures = failures[failures["TURBINE_REG_ID"] == turbine_reg_id]
                failures = failures[
                    (failures["DEFECT_IDENTIFICATION_DATE"] >= date_start)
                    & (failures["DEFECT_IDENTIFICATION_DATE"] < date_end)
                ]
                failures["TS_START"] = failures["DEFECT_IDENTIFICATION_DATE"]
                failures["TS_END"] = failures["ACTION_DATE"]
                failures["TS_END"] = failures["TS_END"].fillna(failures["FAULT_DATE"])
                failures["TS_END"] = failures["TS_END"].fillna("2099-01-01 00:00:00")

                failures["DETECTED_ON_FAIL"] = failures.apply(
                    lambda x: True
                    if x["DEFECT_IDENTIFICATION_DATE"] == x["FAULT_DATE"]
                    else False,
                    axis=1
                )

                failures = failures[
                    ["TS_START", "TS_END", "ALARM_SUBSYSTEM_REG_ID", "ACTION", "DETECTED_ON_FAIL"]
                ]
            return failures
        except Exception as exc:
            raise Exception(
                "Something went wrong when loading the table "
                f"with historical failure data: {exc}"
            ) from exc

    def load_all_failures(
        self, date_start: str, date_end: str, extra_clauses: list[str] = ["1=1"]
    ) -> pd.DataFrame:
        try:
            extra_clauses = ["and " + clause for clause in extra_clauses]
            extra_clauses = " ".join(extra_clauses)

            query = (
                "select TURBINE_REG_ID, DEFECT_IDENTIFICATION_DATE, "
                "FAULT_DATE, ACTION_DATE, ACTION, ALARM_SUBSYSTEM_REG_ID "
                "from major_component_failure "
                "where 1=1 "
                "and IS_MANUFACTURING_ISSUE = 0 " + extra_clauses
            )
            failures = self.load_something(query)
            if not failures.empty:
                failures = failures[
                    (failures["DEFECT_IDENTIFICATION_DATE"] >= date_start)
                    & (failures["DEFECT_IDENTIFICATION_DATE"] < date_end)
                ]
                failures["TS_START"] = failures["DEFECT_IDENTIFICATION_DATE"]
                failures["TS_END"] = failures["ACTION_DATE"]
                failures["TS_END"] = failures["TS_END"].fillna(failures["FAULT_DATE"])
                failures["TS_END"] = failures["TS_END"].fillna("2099-01-01 00:00:00")

                failures["DETECTED_ON_FAIL"] = failures.apply(
                    lambda x: True
                    if x["DEFECT_IDENTIFICATION_DATE"] == x["FAULT_DATE"]
                    else False,
                    axis=1
                )

                failures = failures[
                    ["TURBINE_REG_ID", "TS_START", "TS_END", "ALARM_SUBSYSTEM_REG_ID", "DETECTED_ON_FAIL"]
                ]
            return failures
        except Exception as exc:
            raise Exception(
                "Something went wrong when loading the complete "
                f"table with historical failure data: {exc}"
            ) from exc

    def load_downtimes(
        self, turbine_reg_id: int, date_start: str, date_end: str
    ) -> pd.DataFrame:
        try:
            query = (
                "select CLASSIFICATION_ID, TS_START, TS_END "
                "from downtime_notice with(nolock) "
                f"where TS_START < '{date_end}' "
                f"and TS_END > '{date_start}' "
                f"and TURBINE_REG_ID={turbine_reg_id}"
            )
            data = self.load_something(query)
            return data
        except Exception as exc:
            raise Exception(
                f"Something went wrong loading downtimes from SQL: {exc}"
            ) from exc

    def load_turbines_by_model(
        self,
        turbine_model_reg_ids: list[int] | int,
    ) -> list[int]:
        try:
            if type(turbine_model_reg_ids) == int:
                turbine_model_reg_ids = [turbine_model_reg_ids]

            query = (
                "select TURBINE_REG_ID from turbine_reg "
                "where TURBINE_MODEL_REG_ID in "
                f"({str(turbine_model_reg_ids)[1:-1]})"
            )
            turbine_reg_ids = self.load_something(query)
            return turbine_reg_ids["TURBINE_REG_ID"].tolist()
        except Exception as exc:
            raise Exception(
                "Something went wrong loading turbines by model " f"from SQL: {exc}"
            ) from exc

    def get_channel_subsystem(self, channel_reg_id: int) -> int:
        try:
            query = (
                "SELECT ALARM_SUBSYSTEM_REG_ID "
                "FROM alarm_subsystem_reg, channel_reg "
                "where LOWER(SUBSYSTEM) = LOWER(SUBTYPE) "
                f"and channel_reg_id = {channel_reg_id}"
            )

            subsystem_id = self.load_something(query)

            if subsystem_id.empty:
                raise Exception(f"No results for channel {channel_reg_id}")

            subsystem_id = int(subsystem_id["ALARM_SUBSYSTEM_REG_ID"].iloc[0])

            return subsystem_id

        except Exception as exc:
            raise Exception(
                f"Something went wrong when getting subsystem from SQL: {exc}"
            ) from exc

    def get_generated_alerts(
        self, start_date: str, end_date: str, include_aggregated: bool = False
    ) -> pd.DataFrame:
        try:
            tchala_types = ["CLASSIFIER", "REGRESSOR"]

            coalesce_str = ", ".join([tipo + ".TURBINE_LEVEL" for tipo in tchala_types])
            join_str = " ".join(
                [
                    (
                        f"LEFT JOIN TCHALA_TRAINING_RESULTS_{tipo} {tipo} "
                        f"on alt.TCHALA_TYPE = '{tipo}' "
                        f"AND alt.TCHALA_TRAINING_RESULTS_ID = {tipo}.TCHALA_TRAINING_RESULTS_{tipo}_ID"
                    )
                    for tipo in tchala_types
                ]
            )

            query = (
                "SELECT "
                "TS, TCHALA_TYPE, TCHALA_TRAINING_RESULTS_ID, TURBINE_REG_ID, ALARM_SUBSYSTEM_REG_ID, RELIABILITY, ALERT, "
                f"coalesce({coalesce_str}) as TURBINE_LEVEL "
                "FROM tchala_alerts_new alt "
                f"{join_str} "
                f"WHERE TS >= '{start_date}' AND TS < '{end_date}'"
            )

            if not include_aggregated:
                query += " AND TCHALA_TYPE not like 'AGGREGATED'"

            predictions = self.load_something(query)

            return predictions

        except Exception as exc:
            raise Exception(
                f"Something went wrong when getting alerts from SQL: {exc}"
            ) from exc
