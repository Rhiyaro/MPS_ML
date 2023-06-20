# %%
# Imports

from os import makedirs
import datetime as dt

import src.my_logging as mylog
from src.data_io.data_io import DataIO
from src.manager_manual import ManagerManual
from src.sql_connection import SQLConnection  # pylint: disable=unused-import
from src.tchala import Tchala
from src.tchala_classifier.tchala_classifier import TchalaClassifier
from src.tchala_regressor.tchala_regressor import TchalaRegressor

# from src.tchala_regressor.tchala_regressor import TchalaRegressor

# %%
if __name__ == "__main__":
    # Definitions
    mylog.setup_loggers(root="TrainTchalas")
    mylog.INFO_LOGGER.info("Started the training")
    print("Training Tchalas...")

    save_folder = "output\\results5"
    makedirs(save_folder, exist_ok=True)

    sql_connection = SQLConnection("BR")
    data_io = DataIO.create(
        sql_connection=sql_connection,
        load_from_sql=True,
        save_to_sql=True,
        main_folder=save_folder,
    )

    input_chans = [
        1,
        2,
        3,
        # 5,
        # 6,
        7,
        8,
        9,
        11,
        13,
        16,
        18,
        19,
        20,
        25,
        26,
        28,
        29,
        31,
        32,
        53,
    ]

    tchalas = list()

    # Classifiers
    # tchalas.append(
    #     TchalaClassifier(
    #         level=Tchala.LEVEL_MODEL,
    #         input_chans=input_chans,
    #         folder=save_folder,
    #         filter_inputs=True,
    #         sql_connection=sql_connection,
    #     )
    # )
    # tchalas.append(
    #     TchalaClassifier(
    #         level=Tchala.LEVEL_ALL,
    #         input_chans=input_chans,
    #         folder=save_folder,
    #         filter_inputs=True,
    #         sql_connection=sql_connection,
    #     )
    # )

    # Regressors
    target_channels = [11, 13, 19, 20, 31, 32]
    tchalas.append(
        TchalaRegressor(
            target_channels=target_channels,
            turbine_level=Tchala.LEVEL_TURBINE,
            sql_connection=sql_connection,
        )
    )
    # tchalas.append(
    #     TchalaRegressor(target_channels=target_channels, turbine_level=Tchala.LEVEL_MODEL, sql_connection=sql_connection)
    # )
    # tchalas.append(
    #     TchalaRegressor(target_channels=target_channels, turbine_level=Tchala.LEVEL_ALL, sql_connection=sql_connection)
    # )

    params_manual = {
        "train_date_start": "2019-01-01",
        "train_date_end": "2022-09-01",
    }

    manager = ManagerManual(
        data_io,
        tchalas,
        params_manual,
        with_defects_only=True,
        with_defects_only_clauses=["ALARM_SUBSYSTEM_REG_ID in (4,13)"],
    )

    # %%
    # Training

    # manager.train_all_turbines(individual_turbines=True)
    manager.train_turbine(turbine_reg_id=60520, individual_turbines=True)

    mylog.INFO_LOGGER.info("All trainings done")
    print("Done")
