# import datetime
# import logging
# import os
from abc import ABC, abstractmethod

import pandas as pd
import os
from json import loads
import datetime as dt

from tensorflow import keras
from keras.models import Sequential, load_model

from src.data_io.loader_from_file import LoaderFromFile
from src.data_io.loader_from_sql import LoaderFromSQL
from src.sql_connection import SQLConnection

from keras.optimizers import Adam
from keras import metrics


class LoaderTchalaClassifier(ABC):

    @classmethod
    def create(cls, sql_connection: SQLConnection = None):
        if sql_connection is None:
            return LoaderFromFileTchalaClassifier()
        return LoaderFromSQLTchalaClassifier(sql_connection)

    @abstractmethod
    def load_training_results(
        self,
        turbine_reg_id: int,
        turbine_level: str,
        path: str,
        oneout: bool,
    ) -> pd.DataFrame:
        pass

    # !!!: Keepping structure for compatibility
    def load_list_of_needed_channels(self, turbine_reg_id: int, mode: str) -> list[str]:
        if mode == "train":
            return self.load_list_of_needed_channels_train()
        return self.load_list_of_needed_channels_predict(turbine_reg_id)

    def load_list_of_needed_channels_train(self) -> list[int]:
        inputs = [
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            11,
            13,
            16,
            18,
            19,
            20,
            # 21,
            26,
            28,
            29,
            31,
            32,
            # 43,
            # 44,
            # 45,
            53,
        ]
        auxiliary = [27, 50]

        total = inputs.extend(auxiliary)
        total.sort()
        return total

    @abstractmethod
    def load_list_of_needed_channels_predict(
        self,
        turbine_reg_id: int,
    ) -> list[int]:
        pass

    def load_dates_train(self) -> list[str]:
        date_start = "1900-01-01"

        today = dt.datetime.today()
        date_end = today - dt.timedelta(days=30)
        date_end = str(date_end.date())

        return [date_start, date_end]

    @abstractmethod
    def load_dates_predict(self, level: str, turbine_reg_id: int) -> list[str]:
        pass

    def load_keras_model(
        self,
        turbine_reg_id: int,
        turbine_level: str,
        time_run: dt.datetime,
        oneout: bool = False,
        load_folder: str = "output\\test_runs",
    ) -> Sequential:
        folderpath = f"{load_folder}\\classifier\\{turbine_level}\\"

        path_end = time_run.strftime("%Y-%m-%d_%Hh%Mm%Ss")

        if oneout:
            path_end = f"oneout_{turbine_reg_id}_{path_end}"

        model = load_model(folderpath + path_end, compile=False)

        # Recompiling model
        optimizer = Adam()

        metricas = [
            metrics.CategoricalAccuracy(),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=metricas
        )

        return model


class LoaderFromFileTchalaClassifier(LoaderTchalaClassifier, LoaderFromFile):
    def __init__(self):
        pass

    def load_training_results(
        self,
        turbine_reg_id: int,
        turbine_level: str,
        path: str,
        oneout: bool,
    ) -> dict:
        file = "{path}\\{turbine_level}\\tchala_training_results_classifier.csv"

        try:
            data = self.load_something(file)
            data["TCHALA_TRAINING_RESULTS_CLASSIFIER_ID"] = data.index

            data = data.query("TURBINE_LEVEL == @turbine_level")

            turbine_model_reg_id = self.load_turbine_model(turbine_reg_id)

            if oneout:
                oneout_test_str = f"'%\"oneout_turbine\": {turbine_reg_id}%'"
                data["POSSIBLE"] = data.apply(
                    lambda x: "oneout" in x["METRICS"],
                    axis=1,
                )
                data = data.query("POSSIBLE == True")

            # Get only results where the model appears
            data["POSSIBLE"] = data.apply(
                lambda x: str(turbine_model_reg_id)
                in x["TURBINE_MODEL_REG_IDS"].split("_"),
                axis=1,
            )
            data = data.query("POSSIBLE == True")

            data = data.iloc[0]

            time_run = data["TS_RUN"]

            keras_model = self.load_keras_model(
                turbine_reg_id=turbine_reg_id,
                turbine_level=turbine_level,
                time_run=time_run,
                load_folder=path,
            )

            training_results = {"table_results_training": data, "model": keras_model}

            return training_results

        except Exception as exc:
            raise Exception(
                "Something went wrong loading past results from " f"{file} : {exc}"
            ) from exc

    # TODO: Implement properly
    def load_list_of_needed_channels_predict(
        self,
        turbine_reg_id: int,
    ) -> list[int]:
        return self.load_list_of_needed_channels_train()

    def load_dates_predict(self, level: str, turbine_reg_id: int) -> list[str]:
        raise NotImplementedError(
            "When loading from local files, manual "
            "dates for start and end must be set."
        )


class LoaderFromSQLTchalaClassifier(LoaderTchalaClassifier, LoaderFromSQL):
    def __init__(self, sql_connection: SQLConnection) -> None:
        LoaderFromSQL.__init__(self, sql_connection)

    def load_training_results(
        self,
        turbine_reg_id: int,
        turbine_level: str,
        path: str,
        oneout: bool,
    ) -> pd.DataFrame:
        try:
            if oneout:
                oneout_condition = (
                    f"AND METRICS like '%\"oneout_turbine\": {turbine_reg_id}%'"
                )
            else:
                oneout_condition = r"AND METRICS not like '%oneout%'"

            query = (
                "select * "
                "from tchala_training_results_classifier "
                f"where TURBINE_LEVEL='{turbine_level}' {oneout_condition} "
                "order by TS_RUN desc"
            )

            data = self.load_something(query)

            turbine_model_reg_id = self.load_turbine_model(turbine_reg_id)

            # data = data.query("TURBINE_LEVEL == @turbine_level")

            # Get only results where the model appears
            data["POSSIBLE"] = data.apply(
                lambda x: str(turbine_model_reg_id)
                in x["TURBINE_MODEL_REG_IDS"].split("_"),
                axis=1,
            )
            data = data.query("POSSIBLE == True")
            data = data.iloc[0]

            time_run = data["TS_RUN"]

            keras_model = self.load_keras_model(
                turbine_reg_id=turbine_reg_id,
                turbine_level=turbine_level,
                time_run=time_run,
                load_folder=path,
                oneout=oneout
            )

            training_results = {"table_results_training": data, "model": keras_model}

            return training_results

        except Exception as exc:
            raise Exception(
                "Something went wrong loading classifier training "
                f"results from DB : {exc}"
            ) from exc

    def load_list_of_needed_channels_predict(
        self,
        turbine_reg_id: int,
    ) -> list[int]:
        try:
            query = (
                "select TURBINE_MODEL_REG_IDS, TS_RUN, INPUTS "
                "from tchala_training_results_classifier "
                "order by ts_run desc"
            )
            data = self.load_something(query)
            self.sql_connection.engine.dispose()

            turbine_model = LoaderFromSQL.load_turbine_model(turbine_reg_id)

            model_sets = pd.unique(data["TURBINE_MODEL_REG_IDS"])
            channels = set()
            for model_set in model_sets:
                if str(turbine_model) in model_set.split("_"):
                    channels.update(
                        loads(
                            data.query("TURBINE_MODEL_REG_IDS == @model_set")
                            .query("TS_RUN == TS_RUN.max()")["INPUTS"]
                            .iloc[0]
                        )
                    )

            return list(channels)

        except Exception as exc:
            raise Exception(
                "Something went wrong loading list of needed "
                f"channels from SQL: {exc}"
            ) from exc

    def load_dates_predict(self, level: str, turbine_reg_id: int) -> list[str]:
        return super().load_dates_predict(level, turbine_reg_id)
