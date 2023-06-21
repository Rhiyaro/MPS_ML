from abc import ABC, abstractmethod

import pandas as pd


class MLModel(ABC):
    # Constants
    LEVEL_PANEL = "panel"
    LEVEL_MODEL = "model"
    LEVEL_ALL = "all"

    MODE_TRAINING = "train"
    MODE_PREDICT = "predict"
    MODE_ONEOUT_TRAINING = "oneout_training"

    ALERTS_TABLE = "detection_alerts"

    def __init__(self, model_level: str, oneout: bool = False) -> None:
        self.model_level = model_level
        self.oneout = oneout

    @abstractmethod
    def train(self, data: pd.DataFrame, failures: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame, turbine_reg_id: int) -> None:
        pass

    @abstractmethod
    def compile_results(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def select_possible_channels(
            self, data: pd.DataFrame, initial_channels: list[int], threshold: float = 0.25
    ) -> tuple[list, list]:
        # DONE: Remove inputs that:
        # [X]Don't appear in all turbines
        # [X]Have more than a threshold of NaNs percentage

        remove_set = set()
        turbines = list(set(data["TURBINE_REG_ID"].to_list()))

        for turbine in turbines:
            data_turbine = data.loc[data["TURBINE_REG_ID"] == turbine].copy()

            considerate_inputs = initial_channels.copy()
            for chan in initial_channels:
                if chan not in data_turbine.columns:
                    remove_set.add(chan)
                    considerate_inputs.remove(chan)

            data_turbine = data_turbine[considerate_inputs]
            aux = (data_turbine.isna()).any()

            for col in aux[aux].index:
                if data_turbine[col].isna().sum() / len(data_turbine) > threshold:
                    remove_set.add(col)

        input_set = set(initial_channels)
        input_set = input_set - remove_set

        possible_channels = list(input_set)
        possible_channels.sort()

        remove_list = list(remove_set)
        remove_list.sort()

        return possible_channels, remove_list

    def generate_continuous_alarms(self, alerts_df: pd.DataFrame):
        alerts_df2 = alerts_df.copy()

        window_days = 14
        alert_threshold = 10 / window_days

        window = window_days  # f"{window_days}D"

        alerts: pd.Series = alerts_df2["ALERT"]

        continuous = alerts.rolling(window=window).mean()

        continuous_alerts = (continuous >= alert_threshold).astype(int)

        alerts_df2.loc["ALERT"] = continuous_alerts

        return alerts_df2
