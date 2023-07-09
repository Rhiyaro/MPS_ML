import datetime as dt

import pandas as pd

import src.my_logging as mylog
import src.data_treatment as data_treatment
from src.data_io.data_io import DataIO
from src.ml_model import MLModel


class Manager:
    def __init__(
            self,
            data_io: DataIO,
            models: list[MLModel],
            params: dict,
            with_defects_only: bool = None,
            with_defects_only_clauses: list[str] = None,
            clean_all_nans: bool = False,
            usable_channels: list[int] = None
    ) -> None:
        self.data_io = data_io
        self.models = models

        self.clean_all_nans = clean_all_nans

        # Check if there is a Regressor to be trained. If not, can limit the
        # fetched turbines data to only those which have past failures
        if with_defects_only is not None:
            self.with_defects_only = with_defects_only
            self.with_defects_only_clauses = with_defects_only_clauses

        if usable_channels is None:
            self.usable_channels = [1, 2, 3, 4]

        self.params = params

    def load_dates_train(self, panel_id):
        # TODO: Loads based on model definition
        raise NotImplemented()

    def load_dates_predict(self, panel_id):
        # TODO: Loads based on model definition
        raise NotImplemented()

    def load_panels_model_sets(self):
        return [[1], [2]]

    def train_all_panels(
            self, complete_set: bool = True, individual_panels: bool = True, individual_models: bool = True
    ) -> list[int]:
        mylog.INFO_LOGGER.info("Starting all panels")
        panel_model_sets = self.load_panels_model_sets()
        data_all = []
        failures_all = []
        for panel_model_set in panel_model_sets:
            data, failures = self.train_panel_model(
                panel_model_id=panel_model_set,
                individual_panels=individual_panels,
                individual_models=individual_models,
            )
            if complete_set:
                data_all.append(data)
                failures_all.append(failures)

        if not complete_set:
            return []

        mylog.INFO_LOGGER.info("Training for all panels at once started")
        data_all = pd.concat(data_all, axis=0)
        failures_all = pd.concat(failures_all, axis=0)
        self.train_general(
            [tch for tch in self.models if tch.model_level == MLModel.LEVEL_ALL],
            data_all,
            failures_all,
            training_level=MLModel.LEVEL_ALL,
        )

        turbines_trained = list(data_all["TURBINE_REG_ID"].unique())
        turbines_trained.sort()

        return turbines_trained

    def train_panel_model(
            self,
            panel_model_id: list[int] | int,
            individual_panels: bool = True,
            individual_models: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mylog.INFO_LOGGER.info(f"Starting panels of model(s) {panel_model_id}")

        panel_ids = self.data_io.loader.load_panel_by_model_set(
            panel_model_id
        )
        data_model = []
        failures_model = []

        if self.with_defects_only:
            date_start = self.params["train_date_start"]
            date_end = self.params["train_date_end"]

            extra_clauses = self.with_defects_only_clauses

            all_failures = self.data_io.loader.load_all_failures(
                date_start,
                date_end,
                extra_clauses=extra_clauses,
            )
            failure_panels = all_failures["panel_id"].to_list()
            panel_ids = list(
                set(panel_ids).intersection(set(failure_panels))
            )

        panel_ids.sort()
        for panel_id in panel_ids:
            data, failures = self.train_panel(panel_id, individual_panels)
            if individual_models:
                data_model.append(data)
                failures_model.append(failures)

        if not individual_models:
            return pd.DataFrame(), pd.DataFrame()
        mylog.INFO_LOGGER.info(
            f"Training for turbine model(s) {panel_model_id} started"
        )
        data_model = pd.concat(data_model, axis=0)
        failures_model = pd.concat(failures_model, axis=0)

        if individual_models:
            self.train_general(
                [t for t in self.models if t.model_level == MLModel.LEVEL_MODEL],
                data_model,
                failures_model,
                training_level=MLModel.LEVEL_MODEL,
            )

        return data_model, failures_model

    def train_panel(
            self, panel_id: int, individual_panels: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mylog.INFO_LOGGER.info(f"Training for turbine {panel_id} started")
        data_panel = pd.DataFrame()
        failures = pd.DataFrame()
        try:
            date_start, date_end = self.load_dates_train(panel_id)

            # Limiting date_end to 1 month before
            date_end = min(
                [
                    date_end,
                    str(
                        dt.datetime.combine(dt.datetime.now(), dt.time.min)
                        - dt.timedelta(days=30, seconds=1)
                    ),
                ]
            )

            data_panel = self.prepare_panel_data(
                panel_id,
                date_start,
                date_end,
            )

            failures = self.data_io.loader.load_panel_failures(
                panel_id, date_start, date_end
            )

            if individual_panels:
                self.train_general(
                    models=[t for t in self.models if t.model_level == MLModel.LEVEL_PANEL],
                    data_panel=data_panel,
                    failures=failures,
                    training_level=MLModel.LEVEL_PANEL,
                )
        except Exception as exc:
            mylog.INFO_LOGGER.error(
                "Something went wrong when training the panel "
                f"{panel_id}: {exc}"
            )
            mylog.EXC_LOGGER.exception(exc)
            data_panel = pd.DataFrame()
            failures = pd.DataFrame()
        finally:
            return data_panel, failures

    def predict_panel(self, panel_id: int):
        mylog.INFO_LOGGER.info(f"Prediction for panel {panel_id} started")
        try:
            date_start, date_end = self.load_dates_predict(panel_id)

            data_turbine = self.prepare_panel_data(
                panel_id=panel_id,
                date_start=date_start,
                date_end=date_end,
            )

            data_turbine = data_turbine.sort_values(by="TS")

            for mlmodel in self.models:
                mylog.INFO_LOGGER.info(f"{mlmodel.name} - Start")
                try:
                    mlmodel.predict(data_turbine, panel_id)
                    mlmodel.compile_results()

                    self.data_io.saver.save_results(mlmodel.results)
                    mlmodel.reset()
                except Exception as exc:
                    mylog.INFO_LOGGER.error(exc)
                    mylog.EXC_LOGGER.exception(exc)
                mylog.INFO_LOGGER.info(f"{mlmodel.name} - All done")
        except Exception as exc:
            mylog.INFO_LOGGER.error(
                "Something went wrong when training the turbine: "
                f"{panel_id}: {exc}"
            )
            mylog.EXC_LOGGER.exception(exc)

    def prepare_panel_data(
            self,
            panel_id: int,
            date_start: str,
            date_end: str,
            remove_based_on_power: bool = False,
    ) -> pd.DataFrame:
        channel_ids = self.usable_channels
        data_panel = self.data_io.loader.load_panel_data(panel_id, channel_ids, date_start, date_end)

        data_panel = data_treatment.remove_spurious_data(data_panel)
        data_panel = data_treatment.remove_full_lines_nan(data_panel)
        data_panel = data_treatment.remove_duplicates(data_panel)
        # data_panel = data_treatment.remove_excessive_zeroes(data_panel)
        # data_panel = data_treatment.remove_frozen(data_panel)
        # data_panel = data_treatment.compensate_ambient_temperature(
        #     data_panel, channel_amb_temp=27
        # )

        if remove_based_on_power:
            # TODO: Add if there is a channel for power
            pass
            # pnom = self.data_io.loader.load_turbine_pnom(panel_id)
            # data_panel = data_treatment.remove_low_power(data_panel, pnom)
            # data_panel = data_treatment.remove_excessive_power(data_panel, pnom)

        data_panel = data_treatment.remove_full_lines_nan(data_panel)

        data_panel["panel_id"] = panel_id
        data_panel["panel_model_id"] = self.data_io.loader.load_panel_model(
            panel_id
        )

        return data_panel

    def train_general(
            self,
            models: list[MLModel],
            data_panel: pd.DataFrame,
            failures: pd.DataFrame,
            training_level: str,
    ) -> None:
        for model in models:
            mylog.INFO_LOGGER.info(f"{model.name} - Start")
            try:
                model.train(data_panel, failures)
                model.compile_results()
                self.data_io.saver.save_results(model.results)
                model.reset()
            except Exception as exc:
                mylog.INFO_LOGGER.error(
                    f"Error during train sequence for model {model.name}: {exc}"
                )
                mylog.EXC_LOGGER.exception(exc)
            mylog.INFO_LOGGER.info(f"{model.name} - All done")
