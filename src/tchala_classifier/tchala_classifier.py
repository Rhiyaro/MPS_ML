#  Imports
import datetime
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import metrics
from keras.layers import Dense, Dropout
from keras.models import Sequential  # , load_model
from keras.optimizers import Adam  # ,RMSprop,
from keras.utils import to_categorical

import src.my_logging as mylog
from src import data_treatment
from src import encoder
from src.tchala import Tchala
from src.sql_connection import SQLConnection
from src.tchala_classifier.loader_tchala_classifier import LoaderTchalaClassifier


#  Class


class TchalaClassifier(Tchala):
    def __init__(
        self,
        input_chans: list[int] = None,
        level: str = Tchala.LEVEL_MODEL,
        folder: str = "output\\test_runs",
        model_type: str = "keras",
        binary: bool = False,
        fail_only: bool = False,
        filter_inputs: bool = True,
        replicate_fail: bool = False,
        percent_split_test: float = 0.20,
        fail_margin_before: int = 60,
        fail_margin_after: int = 7,
        fail_duration_before: int = 30,
        fail_duration_after: int = 30,
        fail_max_duration: int = None,
        random_state: int = 9,
        sql_connection: SQLConnection = None,
        oneout: bool = False,
    ):
        super().__init__(turbine_level=level, oneout=oneout)

        self.step = "creation"

        # Default, but open to changes
        self.input_chans = input_chans

        self.folder = folder

        self.model_type = model_type
        self.binary = binary
        self.fail_only = fail_only
        self.filter_inputs = filter_inputs
        self.percent_split_test = percent_split_test

        self.fail_margin_before = fail_margin_before
        self.fail_margin_after = fail_margin_after
        self.fail_duration_after = fail_duration_after
        self.fail_duration_before = fail_duration_before
        self.fail_max_duration = fail_max_duration

        self.replicate_fail = replicate_fail
        self.random_state = random_state

        self.loader = LoaderTchalaClassifier.create(sql_connection=sql_connection)

        self.name = "TchalaClassifier"
        if oneout:
            self.name += "OneOut"

        self.target_col = "FAIL_CODE"

        self.results = {}

    def load_classifier(self, turbine_reg_id: int) -> tuple[pd.DataFrame, keras.Model]:
        level = (
            self.turbine_level
            if self.turbine_level != Tchala.LEVEL_TURBINE
            else str(turbine_reg_id)
        )

        loaded = self.loader.load_training_results(
            turbine_reg_id=turbine_reg_id,
            turbine_level=level,
            path=self.folder,
            oneout=self.oneout,
        )
        training_results = loaded["table_results_training"]
        model = loaded["model"]
        return training_results, model

    def train(self, data: pd.DataFrame, failures: pd.DataFrame):
        train_data = data.copy()
        failures_train = failures.copy()

        self.step = "defects"
        failures_train = self.filter_failures(failures_train)

        turbines = list(set(failures_train["TURBINE_REG_ID"].to_list()))

        assert (
            self.oneout == False or len(turbines) > 1
        ), "For OneOut training, data dataframe should contain data from more than one turbine"

        # Filtering data to only turbines with fails
        train_data = train_data.loc[train_data["TURBINE_REG_ID"].isin(turbines)]

        if self.oneout:
            final_results = {
                "mode": Tchala.MODE_ONEOUT_TRAINING,
                "oneout_trainings": list(),
            }
            for turbine in turbines:
                oneout_data = train_data.query("TURBINE_REG_ID != @turbine").copy()
                oneout_turbines = turbines.copy()
                oneout_turbines.remove(turbine)

                results = self.train_dataframe(
                    turbines=oneout_turbines,
                    train_data=oneout_data,
                    failures_train=failures_train,
                )
                del results["mode"]

                results["ONEOUT_TURBINE"] = turbine
                metrics = json.loads(results["METRICS"])
                metrics["oneout_turbine"] = turbine
                results["METRICS"] = json.dumps(metrics)

                final_results["oneout_trainings"].append(results)

        else:
            final_results = self.train_dataframe(
                turbines=turbines, train_data=train_data, failures_train=failures_train
            )

        self.results = final_results

        # mylog.INFO_LOGGER.info(f"{self.name} - {self.level}-{turbine_models}: Done")

    def train_dataframe(
        self,
        turbines: list[int],
        train_data: pd.DataFrame,
        failures_train: pd.DataFrame,
    ) -> dict:
        # Filter inputs
        target_inputs = self.input_chans.copy()

        # target_inputs = self.define_inputs(train_data)
        target_inputs, _ = self.select_possible_channels(
            data=train_data, initial_channels=self.input_chans
        )
        # train_data = train_data[train_data.columns.intersection(target_inputs)]

        # Filtering NaN data
        # threshold = (1/3)
        # data = data.loc[(data[self.input_chans].isnull().sum(axis=1)
        #                  < len(self.input_chans)*threshold)]

        # Labeling data
        self.step = "label"
        labeled_data = list()
        for turbine in turbines:
            labeled_data.append(
                self.label_data(
                    turbine,
                    failures_train,
                    train_data,
                )
            )

        labeled_data = pd.concat(labeled_data)

        labeled_data = labeled_data.query(f"{self.target_col} != -1")

        target_classes = list(labeled_data[self.target_col].unique())
        target_classes.sort()
        # target_classes.insert(0, 0)  # 0 = nothing
        # self.generate_encoder(target_classes)

        # Replicating failure data

        if self.replicate_fail:
            ratio = round(
                (
                    len(labeled_data.loc[labeled_data[self.target_col].isin([0])])
                    / len(
                        labeled_data.loc[~labeled_data[self.target_col].isin([-1, 0])]
                    )
                )
                * 1
                / 3
            )

            if ratio > 3:
                ratio = 3
            if ratio < 2:
                ratio = 1

            labeled_data = pd.concat(
                [
                    labeled_data,
                    pd.concat(
                        [labeled_data.loc[~labeled_data[self.target_col].isin([-1, 0])]]
                        * ratio
                    ),
                ]
            )

        # Split data train/test
        self.step = "split"
        data_train = list()
        data_test = list()

        # Splitting the percentage for each class
        for fail_code in target_classes:
            split_mask = labeled_data[self.target_col] == fail_code

            if not split_mask.any():
                continue

            data_train_aux, data_test_aux = train_test_split(
                labeled_data.loc[split_mask],
                test_size=self.percent_split_test,
                random_state=self.random_state,
            )

            data_train.append(data_train_aux)
            data_test.append(data_test_aux)

        data_train = pd.concat(data_train)
        data_test = pd.concat(data_test)

        # Separating in X and Y
        x_train = data_train.loc[:, target_inputs].copy()
        x_test = data_test.loc[:, target_inputs].copy()

        y_train = data_train[self.target_col]
        y_test = data_test[self.target_col]

        # Normalize
        self.step = "normalize"
        scaler = data_treatment.calculate_mean_std(x_train)
        x_train = data_treatment.normalize(x_train, scaler)
        x_test = data_treatment.normalize(x_test, scaler)

        # Sorting columns
        self.step = "formatting"
        x_train = x_train.sort_index(axis=1)
        x_test = x_test.sort_index(axis=1)

        # Filling NaNs
        x_train = x_train.fillna(0)
        x_test = x_test.fillna(0)

        # Formatting for Keras
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        label_enc = self.generate_encoder(target_classes)

        y_train = label_enc.transform(y_train)
        y_test = label_enc.transform(y_test)

        y_train = to_categorical(y_train, len(target_classes))
        y_test = to_categorical(y_test, len(target_classes))

        # Configuring model
        self.step = "modelConfig"
        model = self.configure_model(
            target_inputs=target_inputs, target_classes=target_classes
        )

        # Train
        self.step = "train"
        batch_size = 32
        epochs = 10
        workers = 4
        multi = True

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test, y_test),
            workers=workers,
            use_multiprocessing=multi,
        )

        # Evaluation
        self.step = "eval"
        evaluation = model.evaluate(
            x=x_test,
            y=y_test,
            verbose=0,
            workers=workers,
            use_multiprocessing=multi,
            return_dict=True,
        )

        # Report results
        self.step = "buildResults"

        metrics = {k: round(v, 4) for k, v in evaluation.items()}

        metrics = json.dumps(metrics)

        turbine_models = list(train_data["TURBINE_MODEL_REG_ID"].unique())
        turbine_models.sort()
        turbine_models = "_".join(map(str, turbine_models))
        # inputs = '_'.join(map(str, target_inputs))
        inputs = str(target_inputs)

        mylog.INFO_LOGGER.info(f"{self.name} report: {metrics}")

        results = {
            "mode": Tchala.MODE_TRAINING,
            "TURBINE_LEVEL": self.turbine_level,
            "TURBINE_MODEL_REG_IDS": turbine_models,
            "INPUTS": inputs,
            "SCALER_DATA": scaler,
            "LABEL_DATA": label_enc,
            "METRICS": metrics,
            "TS_START": str(min(train_data.index)),
            "TS_END": str(max(train_data.index)),
            "TS_RUN": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "MODEL": model,
        }

        return results

    def filter_failures(self, failures: pd.DataFrame) -> pd.DataFrame:
        # Don't considerate some failures:
        # [ ]with less than a minimum number of component occurrences,
        # [X]before effective time,
        # [X]less than a minimum occurrence period,
        # [X]that are series defects,
        # [X]action is not 'Replacement'
        # [X]Component (or Subsystem ID) is one that should be ignored (for any reason)

        # Ignore specific components
        target_alarm_ids = [4, 13]
        failures = failures.loc[
            failures["ALARM_SUBSYSTEM_REG_ID"].isin(target_alarm_ids)
        ]

        # Filter by action
        failures = failures.query("ACTION == 'REPLACEMENT'")

        # Filter by to-action time
        if "TS_END" in failures.columns:
            failures = failures.loc[
                (failures["TS_END"] - failures["TS_START"])
                > datetime.timedelta(days=10)
            ]

        if "ACTION_DATE" in failures.columns:
            failures = failures.loc[
                (failures["ACTION_DATE"] - failures["DEFECT_IDENTIFICATION_DATE"])
                > datetime.timedelta(days=10)
            ]

        return failures

    def define_inputs(self, data: pd.DataFrame, threshold=0.25) -> list[int]:
        # DONE: Remove inputs that:
        # [X]Don't appear in all turbines
        # [X]Have more than a threshold of NaNs percentage

        remove_set = set()
        turbines = list(set(data["TURBINE_REG_ID"].to_list()))

        for turbine in turbines:
            data_turbine = data.loc[data["TURBINE_REG_ID"] == turbine].copy()

            considerate_inputs = self.input_chans.copy()
            for chan in self.input_chans:
                if chan not in data_turbine.columns:
                    remove_set.add(chan)
                    considerate_inputs.remove(chan)

            data_turbine = data_turbine[considerate_inputs]
            aux = (data_turbine.isna()).any()

            for col in aux[aux].index:
                if data_turbine[col].isna().sum() / len(data_turbine) > threshold:
                    remove_set.add(col)

        input_set = set(self.input_chans)
        input_set = input_set - remove_set

        target_input_chans = list(input_set)
        target_input_chans.sort()

        return target_input_chans

    def label_data(
        self, turbine_reg_id: int, failures: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        turbine_failures = failures.query("TURBINE_REG_ID == @turbine_reg_id").copy()
        turbine_data = data.query("TURBINE_REG_ID == @turbine_reg_id").copy()

        turbine_data.insert(len(turbine_data.columns), self.target_col, 0)
        today = datetime.date.today()

        for _, row in turbine_failures.iterrows():
            # Defining starts
            fail_start = row["TS_START"]
            fail_remove_margin = self.fail_margin_before

            if row["DETECTED_ON_FAIL"]:
                fail_start = fail_start - datetime.timedelta(
                    days=self.fail_duration_before
                )
                fail_remove_margin = int(fail_remove_margin * 1.5)

            start_margin = fail_start - datetime.timedelta(days=fail_remove_margin)

            # Defining ends
            fail_end = row["TS_END"]
            end_margin = fail_end + datetime.timedelta(days=self.fail_margin_after)

            if row["TS_END"].date() > today:
                fail_end = fail_start + datetime.timedelta(
                    days=self.fail_duration_after
                )

            # Checking max duration
            if self.fail_max_duration is not None and (fail_end - fail_start).days > (self.fail_max_duration):
                fail_start = fail_end - datetime.timedelta(days=self.fail_max_duration)

            turbine_data.loc[
                (turbine_data.index > start_margin) & (turbine_data.index < end_margin),
                self.target_col,
            ] = -1

            turbine_data.loc[
                (turbine_data.index > fail_start) & (turbine_data.index < fail_end),
                self.target_col,
            ] = row["ALARM_SUBSYSTEM_REG_ID"]

        return turbine_data

    def configure_model(
        self, target_inputs: list[int], target_classes: list[int]
    ) -> None:
        input_size = len(target_inputs)

        input_shape = (input_size,)
        num_classes = len(target_classes)
        num_layers = 4

        model = Sequential()

        # Defining layers sizes
        sizes = [int(x) for x in np.linspace(input_size, num_classes, num_layers)]

        model.add(
            Dense(sizes[0], activation="tanh", input_shape=input_shape, name="layer_in")
        )

        # Dropout rate = chance do drop inputs
        model.add(Dropout(0.2, seed=self.random_state * 2))

        model.add(Dense(sizes[1], activation="elu", name="layer1"))

        model.add(Dropout(0.4, seed=self.random_state * 3))

        model.add(Dense(sizes[2], activation="elu", name="layer2"))

        # Output layer
        model.add(Dense(num_classes, activation="softmax", name="layer_out"))

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

    def generate_encoder(
        self,
        target_classes: list[str],
        no_fail_code: int | str = 0,
    ) -> LabelEncoder | None:
        # Encodes ALARM_SUBSYSTEM_REG_ID to sequence start in 0
        # Needed for Keras, since the IDs are also numerical

        if self.binary:
            label_enc = BinaryEncoder(no_fail_code)

        else:
            labels = target_classes

            labels_filtered = list(dict.fromkeys(labels))

            label_enc = LabelEncoder()

            label_enc.fit_transform(labels_filtered)

        return label_enc

    def predict(self, data: pd.DataFrame, turbine_reg_id: int = None) -> None:
        try:
            self.predict_turbine(data=data, turbine_reg_id=turbine_reg_id)
        except Exception as exc:
            mylog.INFO_LOGGER.error(
                f"{self.name}.predict({turbine_reg_id}) - " f"{self.step}:{exc}"
            )
            mylog.EXC_LOGGER.exception(exc)
            raise

    def predict_turbine(self, data: pd.DataFrame, turbine_reg_id: int) -> None:
        self.step = "start"
        data_predict = data.copy()

        # Loading model
        self.step = "load_training"
        turbine_model_reg_id = self.loader.load_turbine_model(turbine_reg_id)
        training_results, model = self.load_classifier(turbine_reg_id=turbine_reg_id)

        # assert set(input_chans).issubset(
        #     set(data.columns)
        # ), "Data features are not compatible with trained model"

        # Getting ID
        classifier_id = training_results["TCHALA_TRAINING_RESULTS_CLASSIFIER_ID"]

        # Load Scaler
        self.step = "load_scaler"
        scaler = encoder.decode_scaler(training_results["SCALER_DATA"])

        # Load Encoder
        self.step = "load_label"
        label_encoder = encoder.decode_label(training_results["LABEL_DATA"])

        # Get only input channels
        self.step = "select_channels"
        input_chans = json.loads(training_results["INPUTS"])
        data_predict = data_predict.loc[:, input_chans]

        # Normalize data
        self.step = "load_normalize"
        data_predict = data_treatment.normalize(data_predict, scaler)

        # Fill NaNs
        self.step = "transforming_data"
        data_predict = data_predict.fillna(0)

        # Sorting columns
        data_predict = data_predict.sort_index(axis=1)

        # Transform into numpy array
        data_np = data_predict.to_numpy()

        # Predict
        self.step = "predict"
        prediction = model.predict(data_np, verbose=0)

        prediction = np.around(prediction, decimals=4)

        prediction = pd.DataFrame(
            data=prediction, index=data_predict.index, dtype=float
        )

        # TODO: Check desired results when no turbine data is present in a whole day

        # Transform in subsystem labels
        prediction.columns = label_encoder.inverse_transform(prediction.columns)

        # Apply alert logic
        self.step = "generate_alerts"
        alert_daily = self.generate_alerts(prediction)
        alert_daily = alert_daily.round({"RELIABILITY": 3})

        # Compile Results
        self.step = "build_results_prediction"
        self.results = {
            "mode": Tchala.MODE_PREDICT,
            "TCHALA_TYPE": "CLASSIFIER",
            "TCHALA_TRAINING_RESULTS_ID": classifier_id,
            "TURBINE_REG_ID": turbine_reg_id,
            "PREDICTION": prediction,
            "ALERT_DAILY": alert_daily,
        }

    def generate_alerts(self, prediction: pd.DataFrame):
        prediction_highest = pd.DataFrame(
            prediction.idxmax(axis=1), index=prediction.index, columns=[self.target_col]
        )

        # TODO: Change parameter definitions of place.
        #   Maybe set as Classifier parameter: Define a alert_params attribute
        alerts_daily = (
            prediction_highest.set_index(
                pd.to_datetime(prediction_highest.index).floor("D")
            )
            .groupby(level=0)
            .apply(
                alert_lambda,
                no_fail_code=0,
                inconclusive_code=50,
                set_percent=0.5,
                fail_sum_percent=0.8,
                nominal_count=144,
                min_data=0.5,
            )
        )

        #

        # TODO: Maybe add reliability calculation based on probabilities means
        # daily_reliability = prediction.set_index(pd.to_datetime(prediction.index).floor(
        #     'D')).groupby(level=0).apply(reliability_lambda)

        alerts_df = pd.DataFrame()
        alerts_df["TS"] = alerts_daily.index
        alerts_df["ALARM_SUBSYSTEM_REG_ID"] = alerts_daily["CODE"].values
        alerts_df["ALERT"] = ((alerts_daily["CODE"] != 0).astype(int)).values
        alerts_df["RELIABILITY"] = alerts_daily["RELIABILITY_DAILY"].values

        return alerts_df

    def compile_results(self):
        mode = self.results["mode"]
        self.step = f"compilingResults{mode}"
        del self.results["mode"]

        match mode:
            case Tchala.MODE_TRAINING:
                results = self.compile_results_training(self.results)

            case Tchala.MODE_PREDICT:
                results = self.compile_results_prediction(self.results)

            case Tchala.MODE_ONEOUT_TRAINING:
                results = dict()
                total_training_results = list()
                for result_dict in self.results["oneout_trainings"]:
                    partial_results = self.compile_results_training(result_dict)

                    total_training_results.append(
                        partial_results["tchala_training_results_classifier"]
                    )
                    # Remove the dataframe from the results, leaving only the model path + model
                    del partial_results["tchala_training_results_classifier"]

                    results.update(partial_results)

                results["tchala_training_results_classifier"] = pd.concat(
                    total_training_results
                )

        self.results = results

    def compile_results_training(self, results: dict = None) -> dict:
        if results is None:
            results = self.results

        model = results["MODEL"]
        del results["MODEL"]

        results["SCALER_DATA"] = encoder.encode_scaler(results["SCALER_DATA"])

        results["LABEL_DATA"] = encoder.encode_label(results["LABEL_DATA"])

        turbine_models = results["TURBINE_MODEL_REG_IDS"]

        time_run = results["TS_RUN"]

        if self.oneout:
            oneout_turbine = results["ONEOUT_TURBINE"]
            del results["ONEOUT_TURBINE"]

        results = {k: [v] for k, v in results.items()}
        results = {
            "tchala_training_results_classifier": pd.DataFrame.from_dict(results)
        }

        # subfolder = "_".join([str(x) for x in turbine_models])
        midfolder = f"{self.turbine_level}"

        subfolder = datetime.datetime.fromisoformat(time_run).strftime(
            "%Y-%m-%d_%Hh%Mm%Ss"
        )
        if self.oneout:
            subfolder = f"oneout_{oneout_turbine}_{subfolder}"

        model_path = f"classifier\\{midfolder}\\{subfolder}"

        results[model_path] = model

        return results

    def compile_results_prediction(self, results: dict = None) -> dict:
        if results is None:
            results = self.results

        table_alert = results["ALERT_DAILY"].copy()
        table_alert["TURBINE_REG_ID"] = results["TURBINE_REG_ID"]
        table_alert["TCHALA_TYPE"] = results["TCHALA_TYPE"]
        table_alert["TCHALA_TRAINING_RESULTS_ID"] = results[
            "TCHALA_TRAINING_RESULTS_ID"
        ]

        results = {Tchala.ALERTS_TABLE: table_alert}

        return results

    def reset(self):
        self.results = {}


#  Binary Encoder Class


class BinaryEncoder:
    def __init__(self, zero_label: str):
        self.zero_label = zero_label
        self.classes_ = [zero_label, "-"]

    def transform(self, labels: list) -> list:
        coded = map(lambda label: 0 if label == self.zero_label else 1, labels)
        return list(coded)


# Alert Lambda Method


def alert_lambda(
    group: pd.Series,
    no_fail_code: float = 0,
    inconclusive_code: float = 50,
    set_percent: float = 0.5,
    fail_sum_percent: float = 0.8,
    nominal_count: float = 144,
    min_data: float = 0.5,
):
    return_dict = dict()

    classes = group.value_counts().to_dict()
    classes = {int(k[0]): v for k, v in classes.items()}

    count = len(group)

    reli = 1

    # If there's less than a minimal amount of data, return as 'no fail'
    if count < nominal_count * min_data:
        return_code = no_fail_code

        return_dict["CODE"] = return_code
        return_dict["RELIABILITY_DAILY"] = round(count / nominal_count, 4)
        return pd.Series(return_dict)

    reli -= 0.1
    # If a class represents at least X percent of the data, set the class
    # X = count * set_percent
    fails_sum = 0
    for code, amount in classes.items():
        if code != 0:
            fails_sum += amount

        if amount > count * set_percent:
            return_code = code

            return_dict["CODE"] = return_code
            return_dict["RELIABILITY_DAILY"] = round(amount / count, 4)
            return pd.Series(return_dict)

    reli -= 0.1
    # If fails summed probabilities is Y or more, return 'inconclusive fail'
    # Y = fail_sum_percent * count
    if fails_sum > fail_sum_percent * count:
        return_code = inconclusive_code

        return_dict["CODE"] = return_code
        return_dict["RELIABILITY_DAILY"] = round(fails_sum / count, 4)
        return pd.Series(return_dict)

    reli -= 0.1
    # In last case, return 'no fail'
    return_code = no_fail_code

    return_dict["CODE"] = return_code
    return_dict["RELIABILITY_DAILY"] = reli
    return pd.Series(return_dict)


def reliability_lambda(group: pd.DataFrame):
    count = len(group)

    daily_reliability = group.sum() / count

    daily_reliability = daily_reliability.round(decimals=4)

    return daily_reliability
