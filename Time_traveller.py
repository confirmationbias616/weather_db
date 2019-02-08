import logging
import sys
import os
import datetime
import random
import pandas as pd
import json

from Wrangle import wrangle
from Train import train
from Predict import predict
from Post_mortem import post_mortem
from meta_hpr import analyze


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"
    )
)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def load_hyperparameters():
    filename = "/Users/Alex/Dropbox (Personal)/hyperparameters.json"
    try:
        with open(filename, "rb") as input_file:
            return json.load(input_file)
    except FileNotFoundError:
        return {
            "iterations": 2,
            "start_date": "2018-10-15",
            "end_date": "2018-10-15",
            "time_span": [20],
            "edge_forecasting": [1],
            "real_time": [0],
            "normalize_data": [1, 0],
            "criterion": ["mse", "mae"],
            "label": "TWN_high",
            "drop_columns": 0,
            "include_only_columns": [["mean_high_T1",  "TWN_high_T1", "EC_high_T1", "longitude", "latitude", "elevation", "rolling_normal_high", "current_temp_T1"]],
            "rolling_average_window": [6],
            "rolling_average_min_periods": [1],
            "TWN_EC_split": 0.7,
            "max_depth": [20],
            "max_features": [5],
            "min_samples_leaf": [4],
            "min_samples_split": [2],
            "n_estimators": [40],
            "cv": [3],
            "precision": [1],
            "date_efficient": 0,
            "region_efficient": [1],
            "exit_on_exception": 1,
        }


def get_datetime(date):
    return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))


hp = load_hyperparameters()

loggr.info("Time Travellin...")

for i in range(hp["iterations"]):
    wrangle_status = 0
    while True:
        try:
            hp = load_hyperparameters()

            if type(hp["start_date"]) is list:
                start_date = get_datetime(random.choice(hp["start_date"]))
                end_date = start_date
            else:
                start_date = get_datetime(hp["start_date"])
                end_date = get_datetime(hp["end_date"])

            try:
                eval_days = int(str(end_date - start_date).split(" ")[0]) + 1
            except ValueError:
                eval_days = 1

            hp_inst = {key: [] for key in list(hp.keys())}
            for item in list(hp_inst.keys()):
                if type(hp[item]) is list:
                    hp_inst[item] = random.choice(hp[item])
                else:
                    hp_inst[item] = hp[item]
            loggr.info(
                "hyperparameters randomly selected for this loop:\n"
                + "".join(["{}:{}\n".format(x, hp_inst[x]) for x in hp_inst])
            )

            MLp_agg, MLa_agg, TWNa_agg, ECa_agg, MLw_agg, TWNw_agg, ECw_agg = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for target_date in [
                str(start_date + datetime.timedelta(days=x)) for x in range(eval_days)
            ]:
                try:
                    loggr.info(
                        "Testing random hyperparameter set {} of {} on date {}".format(
                            i + 1, hp["iterations"], target_date
                        )
                    )
                    wrangle_status = wrangle(
                        target_date=target_date,
                        time_span=hp_inst["time_span"],
                        rolling_average_window=hp_inst["rolling_average_window"],
                        rolling_average_min_periods=hp_inst[
                            "rolling_average_min_periods"
                        ],
                        TWN_EC_split=hp_inst["TWN_EC_split"],
                        date_efficient=hp_inst["date_efficient"],
                        region_efficient=hp_inst["region_efficient"],
                        drop_columns=hp_inst["drop_columns"],
                        include_only_columns=hp_inst["include_only_columns"],
                        label=hp_inst["label"],
                        real_time=hp_inst["real_time"],
                    )
                    if wrangle_status == 1:
                        continue
                    train(
                        target_date=target_date,
                        label=hp_inst["label"],
                        time_span=hp_inst["time_span"],
                        max_depth=hp_inst["max_depth"],
                        max_features=hp_inst["max_features"],
                        min_samples_leaf=hp_inst["min_samples_leaf"],
                        min_samples_split=hp_inst["min_samples_split"],
                        n_estimators=hp_inst["n_estimators"],
                        cv=hp_inst["cv"],
                        precision=hp_inst["precision"],
                        edge_forecasting=hp_inst["edge_forecasting"],
                        normalize_data=hp_inst["normalize_data"],
                        criterion=hp_inst["criterion"],
                    )
                    predict(
                        label=hp_inst["label"],
                        precision=hp_inst["precision"],
                        target_date=target_date,
                        normalize_data=hp_inst["normalize_data"],
                    )
                    MLp, MLa, TWNa, ECa, MLw, TWNw, ECw = post_mortem(
                        target_date=str(
                            get_datetime(target_date) + datetime.timedelta(1)
                        )
                    )
                    MLp_agg.append(MLp)
                    MLa_agg.append(MLa)
                    TWNa_agg.append(TWNa)
                    ECa_agg.append(ECa)
                    MLw_agg.append(MLw)
                    TWNw_agg.append(TWNw)
                    ECw_agg.append(ECw)
                except Exception as e:
                    loggr.exception(
                        "Something went wrong for this date. See next line for details. Skipping date..."
                    )
                    loggr.exception("{e}")
                    if hp["exit_on_exception"]:
                        sys.exit(1)
            if wrangle_status == 1:
                break
            log_time = datetime.datetime.now()
            hp_inst.update(
                {
                    "start_date": start_date,
                    "end_date": end_date,
                    "log_time": log_time,
                    "eval_days": eval_days,
                    "ML_rms": (sum([x ** 2 for x in MLa_agg]) / len(MLa_agg)) ** 0.5,
                    "TWN_rms": (sum([x ** 2 for x in TWNa_agg]) / len(TWNa_agg)) ** 0.5,
                    "EC_rms": (sum([x ** 2 for x in ECa_agg]) / len(ECa_agg)) ** 0.5,
                    "ML_ave": sum(MLa_agg) / len(MLa_agg),
                    "TWN_ave": sum(TWNa_agg) / len(TWNa_agg),
                    "EC_ave": sum(ECa_agg) / len(ECa_agg),
                    "ML win %": sum(MLw_agg) / sum(MLp_agg),
                    "TWN win %": sum(TWNw_agg) / sum(MLp_agg),
                    "EC win %": sum(ECw_agg) / sum(MLp_agg),
                    "points_used": sum(MLp_agg) / len(MLp_agg),
                    "TWN-ML ave": [
                        TWNa_agg[x] - MLa_agg[x] for x in range(len(TWNa_agg))
                    ],
                    "TWN-ML wins": [
                        TWNw_agg[x] - MLw_agg[x] for x in range(len(TWNw_agg))
                    ],
                    "points_used": MLp_agg,
                }
            )
            if wrangle_status == 1:
                break
            try:
                search_results = pd.read_csv(
                    "/Users/Alex/Dropbox (Personal)/HPResults.csv"
                )
            except FileNotFoundError:
                search_results = pd.DataFrame(columns=(["log_time"] + list(hp.keys())))
            search_results = search_results.append(hp_inst, ignore_index=True)
            try:
                search_results.to_csv(
                    "/Users/Alex/Dropbox (Personal)/HPResults.csv", index=False
                )
            except FileNotFoundError:
                loggr.warning("Could not save results!!!")
        except Exception as e:
            loggr.exception("This loop could not finish. Here's why: \n {e}")
            if hp["exit_on_exception"]:
                sys.exit(1)
            loggr.exception("Abandoning this loop and skipping to the next one...")
            continue
        break
