import logging
import requests
import sys
import os
import datetime
import time
from shutil import copyfile
import random
import pandas as pd
import json

from Wrangle import wrangle
from Train import train
from Predict import predict
from Post_mortem import post_mortem


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
        exit_on_exception = False
        with open(filename, "rb") as input_file:
            return json.load(input_file), False
    except FileNotFoundError:
        return (
            {
                "iterations": 2,
                "start_date": "2018-10-15",
                "end_date": "2018-10-15",
                "time_span": [20],
                "edge_forecasting": [1, 0],
                "features": [
                    [
                        "latitude",
                        "longitude",
                        "rolling normal high",
                        "TWN_high_T1",
                        "EC_high_T1",
                        "TWN_high_T1_delta",
                        "EC_high_T1_delta",
                    ]
                ],
                "label": "TWN_high",
                "rolling_average_window": [5],
                "rolling_average_min_periods": [1],
                "max_depth": [20],
                "max_features": [5],
                "min_samples_leaf": [4],
                "min_samples_split": [2],
                "n_estimators": [50],
                "cv": [3],
                "precision": [1],
                "date_efficient": 1,
                "region_efficient": [1],
            },
            True,
        )


def get_datetime(date):
    return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))


hp, exit_on_exception = load_hyperparameters()

loggr.info("Time Travellin...")

for i in range(hp["iterations"]):
    while True:
        try:
            hp = load_hyperparameters()

            start_date = get_datetime(hp["start_date"])
            end_date = get_datetime(hp["end_date"])

            try:
                eval_days = int(str(end_date - start_date).split(" ")[0])
            except ValueError:
                eval_days = 0

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

            ML_agg, TWN_agg, EC_agg, Mean_agg, points_used_agg = [], [], [], [], []
            for target_date in [
                str(start_date + datetime.timedelta(days=x))
                for x in range(eval_days + 1)
            ]:
                try:
                    loggr.info(
                        "Testing random hyperparameter set {} of {} on date {}".format(
                            i + 1, hp["iterations"], target_date
                        )
                    )
                    wrangle(
                        target_date=target_date,
                        time_span=hp_inst["time_span"],
                        rolling_average_window=hp_inst["rolling_average_window"],
                        rolling_average_min_periods=hp_inst[
                            "rolling_average_min_periods"
                        ],
                        date_efficient=hp_inst["date_efficient"],
                        region_efficient=hp_inst["region_efficient"],
                    )
                    points_used = train(
                        target_date=target_date,
                        features=hp_inst["features"],
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
                    )
                    points_used_agg.append(points_used)
                    predict(
                        features=hp_inst["features"],
                        label=hp_inst["label"],
                        precision=hp_inst["precision"],
                        target_date=target_date,
                    )
                    ML, TWN, EC, Mean = post_mortem(target_date=target_date)
                    ML_agg.append(ML)
                    TWN_agg.append(TWN)
                    EC_agg.append(EC)
                    Mean_agg.append(Mean)
                except Exception as e:

                    loggr.exception(
                        "Something went wrong for this date. See next line for details. Skipping date..."
                    )
                    loggr.exception("{e}")
                    if exit_on_exception:
                		sys.exit(1)

            log_time = datetime.datetime.now()
            hp_inst.update(
                {
                    "log_time": log_time,
                    "eval_days": eval_days,
                    "ML_rms": (sum([x ** 2 for x in ML_agg]) / len(ML_agg)) ** 0.5,
                    "TWN_rms": (sum([x ** 2 for x in TWN_agg]) / len(TWN_agg)) ** 0.5,
                    "EC_rms": (sum([x ** 2 for x in EC_agg]) / len(EC_agg)) ** 0.5,
                    "Mean_rms": (sum([x ** 2 for x in Mean_agg]) / len(Mean_agg))
                    ** 0.5,
                    "ML": ML_agg,
                    "points_used": points_used_agg,
                }
            )
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
                pass
        except Exception as e:
            loggr.exception("This loop could not finish. Here's why: \n {e}")
            if exit_on_exception:
                sys.exit(1)
            loggr.exception("Abandoning this loop and skipping to the next one...")
            continue
        break
