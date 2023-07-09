# %% Imports
from os import makedirs
import datetime as dt

from dotenv import dotenv_values

from src.ml_model import MLModel
import src.my_logging as mylog
from src.sql_connection import SQLConnection
from src.data_io.data_io import DataIO
from src.manager_manual import ManagerManual

from src.tchala_classifier.tchala_classifier import MLModelClassifier
from src.tchala_regressor.tchala_regressor import TchalaRegressor

# %% Run
if __name__ == "__main__":

    mylog.setup_loggers()
    mylog.INFO_LOGGER.info("Running test prediction script")
    print("Predicting with Tchalas...")

    # Configurations
    save_folder = "output\\test_runs"
    makedirs(save_folder, exist_ok=True)

    sql_connection = SQLConnection("BR")
    data_io = DataIO.create(
        sql_connection=sql_connection,
        load_from_sql=True,
        save_to_sql=False,
        main_folder=save_folder,
    )

    tchalas = list()
    # Classifiers
    tchalas.append(
        MLModelClassifier(
            level=MLModel.LEVEL_MODEL,
            folder=save_folder,
            sql_connection=sql_connection,
            oneout=True
        )
    )
    # tchalas.append(
    #     TchalaClassifier(
    #         level=Tchala.LEVEL_ALL,
    #         folder=save_folder,
    #         sql_connection=sql_connection,
    #     )
    # )

    # Regressors
    # target_channels = [11, 13, 19, 20, 31, 32]

    # tchalas.append(
    #     TchalaRegressor(
    #         target_channels=target_channels,
    #         turbine_level=Tchala.LEVEL_TURBINE,
    #         sql_connection=sql_connection,
    #     )
    # )
    # tchalas.append(
    #     TchalaRegressor(
    #         target_channels=target_channels,
    #         turbine_level=Tchala.LEVEL_MODEL,
    #         sql_connection=sql_connection,
    #     )
    # )
    # tchalas.append(
    #     TchalaRegressor(
    #         target_channels=target_channels,
    #         turbine_level=Tchala.LEVEL_ALL,
    #         sql_connection=sql_connection,
    #     )
    # )

    # Creating manager
    params_manual = {
        "predict_date_start": "2019-01-01",
        "predict_date_end": "2022-09-01",
    }
    # params_manual = {
    #     "predict_date_start": "2023-01-01",
    #     "predict_date_end": "2023-03-16",
    # }

    manager = ManagerManual(data_io, tchalas, params_manual)

    # Loading turbines
    query = ("SELECT mcf.TURBINE_REG_ID "
             "FROM major_component_failure mcf, turbine_reg tr "
             "WHERE mcf.TURBINE_REG_ID = tr.TURBINE_REG_ID "
             "AND mcf.ACTION like 'REPLACEMENT' "
             "AND mcf.ALARM_SUBSYSTEM_REG_ID in (4, 13) "
             "AND mcf.IS_MANUFACTURING_ISSUE = 0 "
             "AND tr.turbine_model_reg_id in (1, 2, 3, 4, 10, 11) "
             f"AND mcf.DEFECT_IDENTIFICATION_DATE >= '{params_manual['predict_date_start']}' "
             "AND ((ACTION_DATE is not null and DATEDIFF(DAY, DEFECT_IDENTIFICATION_DATE, ACTION_DATE) > 10) or ACTION_DATE is null) "
             )
    predict_turbines = list(manager.data_io.loader.load_something(query)["TURBINE_REG_ID"].unique())
    predict_turbines.sort()

    # Running predictions
    for turbine in predict_turbines:
        manager.predict_panel(turbine, continuous_analysis=True)

    # Generating aggregated alarms
    # manager.generate_aggregated_alarms(
    #     start_day=params_manual["predict_date_start"],
    #     end_day=params_manual["predict_date_end"],
    # )

    print("Done")
