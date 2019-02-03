import pandas as pd
import datetime
import logging
import sys
import numpy as np
import os


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line "
        + "%(lineno)d"
    )
)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def post_mortem(**kwargs):
    try:
        fc_date = kwargs["target_date"]
        if fc_date == str(datetime.datetime.now().date() - datetime.timedelta(1)):
            time_travel = False
        else:
            time_travel = True
    except KeyError:
        fc_date = str(datetime.datetime.now().date() - datetime.timedelta(1))
        time_travel = False

    loggr.info("Predicting for date: {}...".format(fc_date))

    actual_date = str(
        datetime.date(int(fc_date[:4]), int(fc_date[5:7]), int(fc_date[8:]))
        + datetime.timedelta(days=1)
    )
    if time_travel:
        time_travel_string = "time_travel/{} -> ".format(datetime.datetime.now().date())
    else:
        time_travel_string = ""

    dbh = pd.read_csv("/Users/Alex/Coding/weather_db/Data/history_db.csv")
    dbp = pd.read_csv("/Users/Alex/Coding/weather_db/Data/prediction_db.csv")

    dbp = (
        dbp.drop("high", axis=1)
        .merge(
            dbh[dbh.provider == "TWN"][["date", "region", "province", "high"]],
            how="left",
            on=["date", "province", "region"],
        )
        .reset_index()
        .drop("index", axis=1)
    )
    dbp.to_csv("/Users/Alex/Coding/weather_db/Data/prediction_db.csv", index=False)

    dbp.dropna(axis=0, subset=["high", "predictions"], inplace=True)

    dbp["ML_real_diff"] = abs(dbp.high - dbp.predictions)
    dbp["ML_real_diff_r1"] = abs(dbp.high - round(dbp.predictions, 1))
    dbp["ML_real_diff_r0"] = abs(dbp.high - round(dbp.predictions, 0))
    dbp["mean_real_diff"] = abs(dbp.high - dbp.mean_high_T1)
    dbp["TWN_real_diff"] = abs(dbp.high - dbp.TWN_high_T1)
    dbp["EC_real_diff"] = abs(dbp.high - dbp.EC_high_T1)
    dbp["ave_real_diff"] = abs(dbp.high - dbp.rolling_normal_high)
    dbp["ML_1win"] = (dbp["ML_real_diff"] < dbp["TWN_real_diff"]) & (
        dbp["ML_real_diff"] < dbp["EC_real_diff"]
    )
    dbp["TWN_1win"] = (dbp["TWN_real_diff"] < dbp["EC_real_diff"]) & (
        dbp["TWN_real_diff"] < dbp["ML_real_diff"]
    )
    dbp["EC_1win"] = (dbp["EC_real_diff"] < dbp["TWN_real_diff"]) & (
        dbp["EC_real_diff"] < dbp["ML_real_diff"]
    )
    dbp["all_3tie"] = (dbp["EC_real_diff"] == dbp["TWN_real_diff"]) & (
        dbp["EC_real_diff"] == dbp["ML_real_diff"]
    )
    dbp["ML_2tie"] = (
        (
            (dbp["ML_real_diff"] <= dbp["TWN_real_diff"])
            & (dbp["ML_real_diff"] <= dbp["EC_real_diff"])
        )
        & (dbp["all_3tie"] == 0)
        & (dbp["ML_1win"] == 0)
    )
    dbp["TWN_2tie"] = (
        (
            (dbp["TWN_real_diff"] <= dbp["EC_real_diff"])
            & (dbp["TWN_real_diff"] <= dbp["ML_real_diff"])
        )
        & (dbp["all_3tie"] == 0)
        & (dbp["TWN_1win"] == 0)
    )
    dbp["EC_2tie"] = (
        (
            (dbp["EC_real_diff"] <= dbp["TWN_real_diff"])
            & (dbp["EC_real_diff"] <= dbp["ML_real_diff"])
        )
        & (dbp["all_3tie"] == 0)
        & (dbp["EC_1win"] == 0)
    )
    dbp["ML_win"] = dbp["ML_1win"] * 1 + dbp["ML_2tie"] * 0.5 + dbp["all_3tie"] * 0.3
    dbp["TWN_win"] = dbp["TWN_1win"] * 1 + dbp["TWN_2tie"] * 0.5 + dbp["all_3tie"] * 0.3
    dbp["EC_win"] = dbp["EC_1win"] * 1 + dbp["EC_2tie"] * 0.5 + dbp["all_3tie"] * 0.3

    seg_dbp_list = []
    for region in dbp.region.unique():
        seg_dbp_list.append(dbp[dbp.region == region])
    for seg_dbp in seg_dbp_list:
        seg_dbp["ML_real_diff_tr"] = seg_dbp.ML_real_diff.rolling(
            window=14, min_periods=1, center=False
        ).mean()
        seg_dbp["TWN_real_diff_tr"] = seg_dbp.TWN_real_diff.rolling(
            window=14, min_periods=1, center=False
        ).mean()
        seg_dbp["EC_real_diff_tr"] = seg_dbp.EC_real_diff.rolling(
            window=14, min_periods=1, center=False
        ).mean()
    dbp_tr = pd.concat(seg_dbp_list, ignore_index=True)
    dbp = dbp.merge(
        dbp_tr[
            ["date", "region", "ML_real_diff_tr", "TWN_real_diff_tr", "EC_real_diff_tr"]
        ],
        on=["date", "region"],
        how="left",
    )

    ML_reg_diff_mean = dbp.groupby("region")["ML_real_diff"].mean().sort_values()
    TWN_reg_diff_mean = dbp.groupby("region")["TWN_real_diff"].mean().sort_values()
    EC_reg_diff_mean = dbp.groupby("region")["EC_real_diff"].mean().sort_values()
    dbpr = pd.concat([ML_reg_diff_mean, TWN_reg_diff_mean, EC_reg_diff_mean], axis=1)
    dbpr["ML_lowest"] = (dbpr.ML_real_diff < dbpr.TWN_real_diff) & (
        dbpr.ML_real_diff < dbpr.EC_real_diff
    )
    dbpr["TWN_lowest"] = (dbpr.TWN_real_diff < dbpr.ML_real_diff) & (
        dbpr.TWN_real_diff < dbpr.EC_real_diff
    )
    dbpr["EC_lowest"] = (dbpr.EC_real_diff < dbpr.TWN_real_diff) & (
        dbpr.EC_real_diff < dbpr.ML_real_diff
    )

    def decode_lowest(ML_lowest, TWN_lowest, EC_lowest):
        if ML_lowest:
            return "ML"
        elif TWN_lowest:
            return "TWN"
        elif EC_lowest:
            return "EC"
        else:
            return "?"

    dbpr["lowest"] = dbpr.apply(
        lambda row: decode_lowest(row.ML_lowest, row.TWN_lowest, row.EC_lowest), axis=1
    )
    dbpr = dbpr.reset_index().rename(columns={"index": "region"})
    dbpr.to_csv(
        "/Users/Alex/Coding/weather_db/Data/prediction_regional_analysis.csv",
        index=False,
    )

    dbp = dbp.merge(dbpr.reset_index()[["region", "lowest"]], how="left", on="region")
    dbp = dbp.sort_values(["date", "province", "region"])
    dbp.to_csv(
        "/Users/Alex/Coding/weather_db/Data/prediction_db_analysis.csv", index=False
    )

    loggr.info("For entire predictions history:")
    for column in [
        "ML_real_diff",
        "ML_real_diff_r1",
        "ML_real_diff_r0",
        "mean_real_diff",
        "TWN_real_diff",
        "EC_real_diff",
        "ave_real_diff",
    ]:
        loggr.info(
            "\tcolumn {} has an average of {} and an rmse of {}".format(
                column,
                sum(dbp[column]) / len(dbp),
                (sum(dbp[column].apply(lambda x: x ** 2)) / len(dbp)) ** 0.5,
            )
        )

    dbp_latest = dbp[dbp.date == fc_date]
    dbp_latest.to_csv(
        "/Users/Alex/Coding/weather_db/Data/prediction_db_analysis_latest.csv",
        index=False,
    )
    loggr.info("Total points to share: {}".format(len(dbp.ML_win)))
    loggr.info("Total ML wins: {}".format(sum(dbp.ML_win)))
    loggr.info("Total TWN wins: {}".format(sum(dbp.TWN_win)))
    loggr.info("Total EC wins: {}".format(sum(dbp.EC_win)))

    if len(dbp_latest) > 0:
        loggr.info("For yesterday only:")
        for column in [
            "ML_real_diff",
            "ML_real_diff_r1",
            "ML_real_diff_r0",
            "mean_real_diff",
            "TWN_real_diff",
            "EC_real_diff",
            "ave_real_diff",
        ]:
            loggr.info(
                "\tcolumn {} has an average of {} and an rmse of {}".format(
                    column,
                    sum(dbp_latest[column]) / len(dbp_latest),
                    (sum(dbp_latest[column].apply(lambda x: x ** 2)) / len(dbp_latest))
                    ** 0.5,
                )
            )

        # for Time_traveller.py and hyperparametrization only
        MLp = len(dbp_latest.ML_win)
        MLa = sum(dbp_latest.ML_real_diff_r1) / MLp
        TWNa = sum(dbp_latest.TWN_real_diff) / MLp
        ECa = sum(dbp_latest.EC_real_diff) / MLp
        MLw = sum(dbp_latest.ML_win)
        TWNw = sum(dbp_latest.TWN_win)
        ECw = sum(dbp_latest.EC_win)

        loggr.info("Latest points to share: {}".format(len(dbp_latest.ML_win)))
        loggr.info("Latest ML wins: {}".format(sum(dbp_latest.ML_win)))
        loggr.info("Latest TWN wins: {}".format(sum(dbp_latest.TWN_win)))
        loggr.info("Latest EC wins: {}".format(sum(dbp_latest.EC_win)))
    else:
        loggr.warning("No results from yesterday!")

    return MLp, MLa, TWNa, ECa, MLw, TWNw, ECw
