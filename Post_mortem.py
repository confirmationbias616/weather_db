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
        if fc_date == str(datetime.datetime.now().date()):
            time_travel = False
        else:
            time_travel = True
    except KeyError:
        fc_date = str(datetime.datetime.now().date())
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
    pred = pd.read_csv(
        "{}/Predictions/{}{}_predict_tm_high.csv".format(
            PATH, time_travel_string, fc_date
        ),
        dtype={"date": "str"},
    )
    dbh = pd.read_csv("{}/Data/history_db.csv".format(PATH)).drop("time", axis=1)
    dbp = pd.read_csv("{}/Data/prediction_db.csv".format(PATH))

    # Prep history_db and prediction_db to be merged into a glorious results_db
    dbp_fc = dbp[dbp.date==fc_date]
    dbh_fc = dbh[dbh.provider=='TWN'][['date', 'region', 'province', 'high']]
    dbp_fc = dbp_fc.merge(dbh_fc, how='left', on=['date', 'province', 'region'])
    dbp_fc['high'] = np.nan # for now!
    dbp = dbp.append(dbp_fc, ignore_index=False)
    dbp.to_csv("{}/Data/prediction_db.csv".format(PATH), index=False)

    actual = dbh[dbh["date"] == actual_date][["region", "high", "province"]]
    try:
        comp = pred.merge(actual, on=["region", "province"], how="left")
    except KeyError:
        comp = pred.merge(actual, on=["region"], how="left")
    comp["TWN_EC_ave"] = (comp["TWN_high_T1"] + comp["EC_high_T1"]) / 2
    comp["diff_real"] = (comp["high"]) - comp["model_predictions"]
    comp["diff_TWN_rival"] = (comp["high"]) - comp["TWN_high_T1"]
    comp["diff_EC_rival"] = (comp["high"]) - comp["EC_high_T1"]
    comp["diff_mean_rival"] = (comp["high"]) - comp["TWN_EC_ave"]
    # comp = comp.sort_values("diff_real")
    ML_perf = (
        (comp["diff_real"].apply(lambda x: x ** 2).sum()) / len(comp.index)
    ) ** 0.5
    TWN_rival_perf = (
        (comp["diff_TWN_rival"].apply(lambda x: x ** 2).sum()) / len(comp.index)
    ) ** 0.5
    EC_rival_perf = (
        (comp["diff_EC_rival"].apply(lambda x: x ** 2).sum()) / len(comp.index)
    ) ** 0.5
    mean_rival_perf = (
        (comp["diff_mean_rival"].apply(lambda x: x ** 2).sum()) / len(comp.index)
    ) ** 0.5

    comp.to_csv(
        "{}/Predictions/{}{}_compare_actual.csv".format(
            PATH, time_travel_string, actual_date
        )
    )
    loggr.info("ML performance: {}".format(round(ML_perf, 2)))
    loggr.info("TWN rival performance: {}".format(round(TWN_rival_perf, 2)))
    loggr.info("EC rival performance: {}".format(round(EC_rival_perf, 2)))
    loggr.info("Mean rival performance: {}".format(round(mean_rival_perf, 2)))
    f = open(
        "{}/Predictions/{}{}_compare_actual.txt".format(
            PATH, time_travel_string, actual_date
        ),
        "w",
    )
    f.write("ML performance: {}\n".format(round(ML_perf, 2)))
    f.write("TWN rival performance: {}\n".format(round(TWN_rival_perf, 2)))
    f.write("EC rival performance: {}\n".format(round(EC_rival_perf, 2)))
    f.write("Mean rival performance: {}".format(round(mean_rival_perf, 2)))

    return ML_perf, TWN_rival_perf, EC_rival_perf, mean_rival_perf
