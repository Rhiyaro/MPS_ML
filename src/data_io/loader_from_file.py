import pandas as pd

from src.data_io.loader import Loader


class LoaderFromFile(Loader):
    def __init__(self, main_folder: str) -> None:
        self.main_folder = main_folder

        self.path_turbine_reg = "data\\turbine_reg.csv"
        self.path_turbine_model_status = "data\\turbine_model_status.csv"
        self.path_turbine_model_reg = "data\\turbine_model_reg.csv"
        self.path_failures = "data\\tabela_falhas_reg_ids.csv"

    def load_something(self, filename: str) -> pd.DataFrame:
        data = pd.read_csv(filename, sep=";", index_col=0)
        return data

    def load_turbine_data(
        self,
        turbine_reg_id: int,
        channel_reg_ids: list[int],
        date_start: str,
        date_end: str,
    ) -> pd.DataFrame:
        filename = f"data\\{turbine_reg_id}.csv"
        try:
            data = self.load_something(filename)
            data = data[(data.index >= date_start) & (data.index < date_end)]
            data = data[data["CHANNEL_REG_ID"].isin(channel_reg_ids)]
            data = data.reset_index().pivot(
                index="TS", columns="CHANNEL_REG_ID", values="VALUE"
            )
            data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
            return data
        except Exception as exc:
            raise Exception(
                "Something went wrong loading turbine data from " f"the file {filename}"
            ) from exc

    def load_turbine_model(self, turbine_reg_id: int) -> int:
        turbine_reg = self.load_something(self.path_turbine_reg)
        turbine_model_reg_id = turbine_reg.loc[
            turbine_reg["TURBINE_REG_ID"] == turbine_reg_id, "TURBINE_MODEL_REG_ID"
        ].iloc[0]
        return int(turbine_model_reg_id)

    def load_turbine_pnom(self, turbine_reg_id: int) -> int:
        try:
            turbine_model_reg_id = self._load_turbine_model(turbine_reg_id)
            turbine_model_reg = self.load_something(self.path_turbine_model_reg)
            pnom = turbine_model_reg.loc[
                turbine_model_reg["TURBINE_MODEL_REG_ID"] == turbine_model_reg_id,
                "TURBINE_MODEL_REG_PWR",
            ].iloc[0]
            return pnom
        except Exception as exc:
            raise Exception(
                "Something went wrong when loading the turbine " "nominal power"
            ) from exc

    def load_turbine_status_ok(self, turbine_reg_id: int) -> int | None:
        try:
            turbine_model_reg_id = self._load_turbine_model(turbine_reg_id)
            turbine_model_status = self.load_something(self.path_turbine_model_status)
            turbine_model_status = turbine_model_status[
                turbine_model_status["TURBINE_MODEL_REG_ID"] == turbine_model_reg_id
            ]
            status_ok = turbine_model_status.loc[
                turbine_model_status["TURBINE_STATUS_REG_ID"] == 14,
                "TURBINE_STATUS_CODE",
            ].iloc[0]
            return status_ok
        except Exception as exc:
            raise Exception(
                "Something went wrong loading the turbine status " "ok value"
            ) from exc

    def load_table_failures(
        self, turbine_reg_id: int, date_start: str, date_end: str
    ) -> pd.DataFrame:
        try:
            filename = "data\\tabela_falhas_reg_ids.csv"
            failures = self.load_something(filename)
            if not failures.empty:
                failures = failures[failures["TURBINE_REG_ID"] == turbine_reg_id]
                failures = failures[
                    (failures["Defect Identification"] >= date_start)
                    & (failures["Defect Identification"] < date_end)
                ]
                failures["TS_START"] = failures["Defect Identification"]
                failures["TS_END"] = failures["Action Date"]
                failures.loc[failures["TS_END"].isna(), "TS_END"] = failures.loc[
                    failures["TS_END"].isna(), "Failure Date"
                ]
                failures.loc[
                    failures["TS_END"].isna(), "TS_END"
                ] = "2099-01-01 00:00:00"

                failures["DETECTED_ON_FAIL"] = failures.apply(
                    lambda x: True
                    if x["DEFECT_IDENTIFICATION_DATE"] == x["FAULT_DATE"]
                    else False
                )

                failures = failures[["TS_START", "TS_END", "Component", "Action", "DETECTED_ON_FAIL"]]
            return failures
        except Exception as exc:
            raise Exception(
                "Something went wrong when loading the table "
                f"with historical failure data: {exc}"
            ) from exc
        
    def load_all_failures(self, date_start: str, date_end: str) -> pd.DataFrame:
        raise NotImplemented()

    def load_downtimes(
        self, *args, **kwargs
    ) -> pd.DataFrame:  # pylint: disable=unused-argument
        return pd.DataFrame()

    def load_turbines_by_model(
        self,
        turbine_model_reg_ids: list[int],
    ) -> list[int]:
        turbine_reg = self.load_something(self.path_turbine_reg)
        turbine_reg_ids = turbine_reg.loc[
            turbine_reg["TURBINE_MODEL_REG_ID"].isin(turbine_model_reg_ids),
            "TURBINE_REG_ID",
        ]
        return turbine_reg_ids.tolist()

    def get_channel_subsystem(self, channel_reg_id: int) -> int:
        raise NotImplemented()

    def get_generated_alerts(
        self, start_date: str, end_date: str, include_aggregated: bool = False
    ) -> pd.DataFrame:

        filename = f"{self.main_folder}\\tchala_alerts_new.csv"

        try:
            all_predictions = self.load_something(filename)

            all_predictions = all_predictions.query(
                "TS >= @start_date and TS < @end_date"
            )

            if include_aggregated:
                all_predictions = all_predictions.query("TCHALA_TYPE != 'AGGREGATED'")

            return all_predictions

        except Exception as exc:
            raise Exception(
                f"Something went wrong when loading alerts: {exc}"
            ) from exc
