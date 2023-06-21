import datetime as dt

import pandas as pd
from numpy import prod as np_prod

import src.my_logging as mylog
import src.data_treatment as data_treatment
from src import checker
from src.data_io.data_io import DataIO
from src.data_io.saver_to_file import SaverToFile
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
            turbine_level=MLModel.LEVEL_ALL,
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
                turbine_level=MLModel.LEVEL_MODEL,
            )

        return data_model, failures_model

    def train_panel(
            self, panel_id: int, individual_panels: bool = True
    ) -> list[pd.DataFrame]:
        mylog.INFO_LOGGER.info(f"Training for turbine {panel_id} started")
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

            failures = self.data_io.loader.load_table_failures(
                panel_id, date_start, date_end
            )

            failures["TURBINE_REG_ID"] = panel_id

            if individual_panels:
                self.train_general(
                    tchalas=[
                        t
                        for t in self.models
                        if t.model_level == MLModel.LEVEL_TURBINE
                    ],
                    data_turbine=data_panel,
                    failures=failures,
                    turbine_level=MLModel.LEVEL_TURBINE,
                )
        except Exception as exc:
            mylog.INFO_LOGGER.error(
                "Something went wrong when training the turbine "
                f"{panel_id}: {exc}"
            )
            mylog.EXC_LOGGER.exception(exc)
            data_panel = pd.DataFrame()
            failures = pd.DataFrame()
        finally:
            return data_panel, failures  # pylint: disable:lost-exception

    # Test method
    def train_turbine_model_test(
            self,
            turbine_model_reg_id: list[int],
            data_model: pd.DataFrame,
            failures_model: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mylog.INFO_LOGGER.info(f"Starting turbines of model(s) {turbine_model_reg_id}")

        mylog.INFO_LOGGER.info(
            f"Training for turbine model(s) {turbine_model_reg_id} started"
        )

        self.train_general(
            tchalas=[
                tch for tch in self.models if tch.model_level == MLModel.LEVEL_MODEL
            ],
            data_turbine=data_model,
            failures=failures_model,
            turbine_level=MLModel.LEVEL_MODEL,
        )

    def predict_turbine(self, turbine_reg_id: int, continuous_analysis: bool = False):
        mylog.INFO_LOGGER.info(f"Prediction for turbine {turbine_reg_id} started")
        try:
            date_start, date_end = self.load_dates_predict(turbine_reg_id)

            date_start_extended = date_start
            if continuous_analysis:
                date_start_extended = str(
                    dt.date.fromisoformat(date_start) - dt.timedelta(days=13)
                )

            data_turbine = self.prepare_panel_data(
                turbine_reg_id,
                date_start_extended,
                date_end,
                downtime_ids_to_ignore=None,
                remove_based_on_status=True,
                remove_based_on_power=False,
                mode="predict",
            )

            data_turbine = data_turbine.sort_values(by="TS")

            for tchala in self.models:
                mylog.INFO_LOGGER.info(f"{tchala.name} - Start")
                try:
                    tchala.predict(data_turbine, turbine_reg_id)
                    tchala.compile_results()

                    if continuous_analysis:
                        tchala.results[
                            MLModel.ALERTS_TABLE
                        ] = tchala.generate_continuous_alarms(
                            tchala.results[MLModel.ALERTS_TABLE]
                        )
                        tchala.results[MLModel.ALERTS_TABLE] = tchala.results[
                            MLModel.ALERTS_TABLE
                        ].query("TS >= @date_start")

                    self.data_io.saver.save_results(tchala.results)
                    tchala.reset()
                except Exception as exc:
                    mylog.INFO_LOGGER.error(exc)
                    mylog.EXC_LOGGER.exception(exc)
                mylog.INFO_LOGGER.info(f"{tchala.name} - All done")
        except Exception as exc:
            mylog.INFO_LOGGER.error(
                "Something went wrong when training the turbine: "
                f"{turbine_reg_id}: {exc}"
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
            tchalas: list[MLModel],
            data_turbine: pd.DataFrame,
            failures: pd.DataFrame,
            turbine_level: str,
    ) -> None:
        for tchala in tchalas:
            mylog.INFO_LOGGER.info(f"{tchala.name} - Start")
            try:
                tchala.train(data_turbine, failures)
                tchala.compile_results()
                self.data_io.saver.save_results(tchala.results)
                tchala.reset()
                done = True
            except Exception as exc:
                mylog.INFO_LOGGER.error(
                    f"Error during train sequence for TCHALA {tchala.name}: {exc}"
                )
                mylog.EXC_LOGGER.exception(exc)
            mylog.INFO_LOGGER.info(f"{tchala.name} - All done")

    def generate_aggregated_alarms(
            self, start_day: dt.date | str, end_day: dt.date | str
    ):
        try:
            # XXX Temporary, since the calculations for this value must be reviewed
            reliability_threshold = 1  # 0.6
            alert_threshold = 0.5

            predictions = self.data_io.loader.get_generated_alerts(
                start_date=str(start_day), end_date=str(end_day)
            )

            if predictions.empty:
                raise Exception(
                    f"No predictions for period between {start_day} and {end_day}"
                )

            predictions["RELIABILITY"] = predictions["RELIABILITY"].fillna(
                reliability_threshold
            )

            # Get weight of each alerts based on TCHALA Level
            predictions["WEIGHT"] = predictions.apply(get_prediction_weight, axis=1)

            predictions["RELIABILITY_CORRECTED"] = predictions["RELIABILITY"].apply(
                lambda x: 1 if x > reliability_threshold else reliability_threshold
            )

            agg_alerts = list()
            for day in predictions["TS"].unique():
                day_predictions = predictions.query("TS == @day").copy()
                day_predictions = self.aggregate_alerts_for_day(
                    day_predictions=day_predictions,
                    reliability_threshold=reliability_threshold,
                    alert_threshold=alert_threshold,
                )
                agg_alerts.append(day_predictions)

            agg_alerts = pd.concat(agg_alerts)

            agg_alerts = agg_alerts.round({"RELIABILITY": 3})

            # Save new alerts
            self.data_io.saver.save_dataframe(data=agg_alerts, path="tchala_alerts_new")

        except Exception as exc:
            mylog.INFO_LOGGER.error(
                f"Something went wrong when aggregating TCHALA alerts: {exc}"
            )
            mylog.EXC_LOGGER.exception(exc)

    def aggregate_alerts_for_day(
            self,
            day_predictions: pd.DataFrame,
            reliability_threshold: float = 0.6,
            alert_threshold: float = 0.5,
    ) -> pd.DataFrame:
        # Calculate aggregated alert and its information
        agg_alerts = []
        for turbine_reg_id in day_predictions["TURBINE_REG_ID"].unique():
            turbine_alerts = day_predictions.query(
                "TURBINE_REG_ID == @turbine_reg_id"
            ).copy()

            final_alert = (
                    turbine_alerts["ALERT"]
                    * turbine_alerts["WEIGHT"]
                    * turbine_alerts["RELIABILITY_CORRECTED"]
            )
            final_alert = final_alert.sum() / turbine_alerts["WEIGHT"].sum()

            final_alert = 1 if final_alert >= alert_threshold else 0

            subsystem_id = turbine_alerts["ALARM_SUBSYSTEM_REG_ID"].iloc[0]
            if len(turbine_alerts["ALARM_SUBSYSTEM_REG_ID"]) > 1:
                subsystem_id = turbine_alerts.loc[turbine_alerts["WEIGHT"].idxmax()][
                    "ALARM_SUBSYSTEM_REG_ID"
                ]

            reliability = turbine_alerts["RELIABILITY"].sum() / len(turbine_alerts)

            final_alert = {
                "TS": day_predictions["TS"].iloc[0],
                "TURBINE_REG_ID": turbine_reg_id,
                "ALARM_SUBSYSTEM_REG_ID": subsystem_id,
                "ALERT": final_alert,
                "RELIABILITY": reliability,
            }

            agg_alerts.append(final_alert)

        agg_alerts = pd.DataFrame(agg_alerts)

        agg_alerts["TCHALA_TYPE"] = "AGGREGATED"

        return agg_alerts


def get_prediction_weight(row):
    # Based on TCHALA Level and possible other information of the prediciton,
    # get weight of alert

    weights = list()

    # --- Weight based on Turbine Level
    # tchala_type = row["TCHALA_TYPE"]
    prediction_level = row["TURBINE_LEVEL"]

    # Default if no condition matches
    tchala_level = MLModel.LEVEL_ALL

    if prediction_level.isdigit():  # LEVEL_TURBINE or Single Model
        if int(prediction_level) >= 1000:  # Turbine Reg ID
            tchala_level = MLModel.LEVEL_TURBINE

        tchala_level = MLModel.LEVEL_MODEL

    if prediction_level.find("_") != -1:
        tchala_level = MLModel.LEVEL_MODEL

    match tchala_level:
        case MLModel.LEVEL_TURBINE:
            weights.append(4)  # 100% more than 'all'
        case MLModel.LEVEL_MODEL:
            weights.append(3)  # 50% more than 'all'
        case MLModel.LEVEL_ALL:
            weights.append(2)
        case _:
            weights.append(0)

    # --- Weight based on Tchala Type
    tchala_type = row["TCHALA_TYPE"]

    match tchala_type:
        case "REGRESSOR":
            weights.append(1)
        case "CLASSIFIER":
            weights.append(5)
        case _:
            weights.append(0)

    # Final Result
    weight = np_prod(weights)

    return weight
