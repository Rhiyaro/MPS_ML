import datetime as dt

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

# def treat_data(data: pd.DataFrame,
#                status_ok: int | None,
#                pnom: int | None,
#                inputs: list[int],
#                chan_amb_temp: int,
#                chan_other_temps: list[int],
#                downtimes: pd.DataFrame,
#                downtimes_ids_to_ignore: list[int] | None = None
#                ) -> pd.DataFrame:
#     data = remove_full_lines_nan(data)
#     data = remove_duplicates(data)
#     data = remove_frozen(data)
#     data = remove_status_not_ok(data, status_ok)
#     data = remove_low_power(data, pnom)
#     data = remove_downtimes(data, downtimes, downtimes_ids_to_ignore)
#     data = compensate_ambient_temperature(
#         data, chan_amb_temp, chan_other_temps)
#     data = select_inputs(data, inputs)
#     data = remove_full_lines_nan(data)
#     return data


def remove_status_not_ok(data: pd.DataFrame, status_ok: int | None) -> pd.DataFrame:
    if status_ok is not None:
        data = data[data[50] == status_ok]
        data = data.drop(columns=[50])
    return data


def remove_low_power(data: pd.DataFrame, pnom: int | None) -> pd.DataFrame:
    if pnom is not None:
        data = data[data[53] >= (0.05 * pnom)]
    return data


def remove_excessive_power(data: pd.DataFrame, pnom: int | None) -> pd.DataFrame:
    if pnom is not None:
        data = data[data[53] <= (1.05 * pnom)]
    return data


def remove_full_lines_nan(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(how="all")


def remove_excessive_zeroes(data: pd.DataFrame) -> pd.DataFrame:
    skip_channels = []

    drop_channels = ((data == 0).mean() >= 0.5).loc[((data == 0).mean() >= 0.5) == True].index
    drop_channels = [chan for chan in drop_channels if chan not in skip_channels]

    data = data.drop(columns=drop_channels)

    return data


def remove_channels_excessive_nan(data: pd.DataFrame) -> pd.DataFrame:
    return data


def select_inputs(data: pd.DataFrame, inputs: list[int]) -> pd.DataFrame:
    absent_channels = list(set(inputs).difference(set(data.columns)))
    if absent_channels:
        raise Exception(
            "Error when selecting inputs: " f"channels {absent_channels} absent."
        )
    data = data[inputs]
    return data


def compensate_ambient_temperature(
    data: pd.DataFrame, channel_amb_temp: int, channel_other_temps: list[int] = None
) -> pd.DataFrame:
    if channel_amb_temp is None:
        return data
    if channel_amb_temp not in data.columns:
        return data

    if channel_other_temps is None:
        channel_other_temps = [
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            9,
            11,
            12,
            13,
            16,
            18,
            19,
            20,
            21,
            22,
            23,
            26,
            29,
            31,
            32,
            43,
            44,
            45,
        ]

    chans_available = [col for col in data.columns if col in channel_other_temps]
    data[chans_available] = data[chans_available].subtract(
        data[channel_amb_temp], axis=0
    )
    data = data.drop(columns=[channel_amb_temp])
    return data


def remove_frozen(data: pd.DataFrame) -> pd.DataFrame:
    tol = 1e-6
    with np.errstate(invalid="ignore"):
        step = min(abs(data.index[1:] - data.index[:-1]))
        data = data.resample(step).mean()
        bad_1L = data.diff().abs() < tol
        bad_2L = data.diff(2).abs() < tol
        bad_1R = data.diff(-1).abs() < tol
        bad_2R = data.diff(-2).abs() < tol
        has_decimals = (data - data.round()).sum() != 0
        data[bad_1L & bad_2L & has_decimals] = np.nan
        data[bad_1R & bad_2R & has_decimals] = np.nan
    return data


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data[~data.index.duplicated(keep="first")]


def remove_downtimes(
    data: pd.DataFrame, downtimes: pd.DataFrame, ids_to_ignore: list[int] | None = None
) -> pd.DataFrame:
    if not downtimes.empty:
        if ids_to_ignore is not None:
            downtimes = downtimes[~downtimes["CLASSIFICATION_ID"].isin(ids_to_ignore)]
        for _, dwt in downtimes.iterrows():
            data = data[(data.index < dwt["TS_START"]) | (data.index > dwt["TS_END"])]
    return data


def calculate_best_correlation(
    data: pd.DataFrame,
    target_channel: int,
    n_channels: int,
    max_lim_correlation: float,
    min_lim_correlation: float,
) -> list[int]:
    corr = data.corr(numeric_only=True)[[target_channel]]
    corr = corr[corr[[target_channel]] < max_lim_correlation]
    corr = corr[corr[[target_channel]] > min_lim_correlation]
    corr = corr.sort_values(target_channel, ascending=False)
    corr = corr.loc[[row for row in corr.index if row != target_channel]]
    return list(corr.index[:n_channels])


def calculate_metrics(data1: pd.DataFrame, data2: pd.DataFrame) -> dict:
    return {
        "n_samples": data1.shape[0],
        "RMSE": round(metrics.mean_squared_error(data1, data2, squared=True), 2),
        "MAE": round(metrics.mean_absolute_error(data1, data2), 2),
        "MAPE": round(metrics.mean_absolute_percentage_error(data1, data2), 3),
        "R2": round(metrics.r2_score(data1, data2), 3),
    }


def remove_spurious_data(data: pd.DataFrame) -> pd.DataFrame:
    channel_temps = [
        3
    ]
    chans_temps_available = [col for col in data.columns if col in channel_temps]

    data1 = data.copy()
    data1 = data1.fillna(data1.quantile(0.5))

    data2 = data.loc[
        (
            (data1[chans_temps_available] < 120) & (data1[chans_temps_available] > -20)
        ).all(axis=1)
    ].copy()

    data3 = data2.fillna(data2.quantile(0.5))

    qlow = data2.quantile(0.01)
    qhigh = data2.quantile(0.99)
    data4 = data2.loc[((data3 >= qlow) & (data3 <= qhigh)).all(axis=1)]
    return data4


def calculate_mean_std(data: pd.DataFrame) -> list[pd.Series]:
    scaler = [data.mean().round(4), data.std().round(4)]
    return scaler


def normalize(data0: pd.DataFrame, scaler: list[pd.Series]) -> pd.DataFrame:
    data = data0.copy()
    channels = list(set(data.columns) & set(scaler[0].index))
    data = data.loc[:, channels]
    data -= scaler[0].loc[channels]
    data /= scaler[1].loc[channels]
    return data


def unnormalize(data0: pd.DataFrame, scaler: list[pd.Series]) -> pd.DataFrame:
    data = data0.copy()
    channels = list(set(data.columns) & set(scaler[0].index))
    data *= scaler[1].loc[channels]
    data += scaler[0].loc[channels]
    return data


def remove_dates_failures(
    data: pd.DataFrame,
    failures: pd.DataFrame,
    days_before: int = 0,
    days_after: int = 0,
) -> pd.DataFrame:
    for _, row in failures.iterrows():
        days_before_to_remove = days_before
        if row["DETECTED_ON_FAIL"]:
            days_before_to_remove = int(days_before_to_remove * 1.5)

        start_remove = (
            row["TS_START"] - dt.timedelta(days=days_before_to_remove)
        ).strftime("%Y-%m-%d %H:%M:%S")
        end_remove = (row["TS_END"] + dt.timedelta(days=days_after)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        data = data[(data.index <= start_remove) | (data.index >= end_remove)]
    return data


def remove_dates_failures_by_turbine(
    data: pd.DataFrame,
    failures: pd.DataFrame,
    days_before: int = 0,
    days_after: int = 0,
) -> pd.DataFrame:
    clean_data = list()
    for turbine in data["TURBINE_REG_ID"].unique():
        clean_data.append(
            remove_dates_failures(
                data=data.query("TURBINE_REG_ID == @turbine").copy(),
                failures=failures.query("TURBINE_REG_ID == @turbine").copy(),
                days_before=days_before,
                days_after=days_after,
            )
        )

    clean_data = pd.concat(clean_data)

    return clean_data
