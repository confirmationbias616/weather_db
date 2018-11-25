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
            "normalize_data": [1, 0],
            "label": "TWN_high",
            "drop_columns": 0,
            "include_only_columns": 0,
            "rolling_average_window": [5],
            "rolling_average_min_periods": [1],
            "TWN_EC_split": 0.7,
            "max_depth": [20],
            "max_features": [5],
            "min_samples_leaf": [4],
            "min_samples_split": [2],
            "n_estimators": [40],
            "cv": [3],
            "precision": [1],
            "date_efficient": 1,
            "region_efficient": [1],
            "exit_on_exception": 1
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

            ML_agg, TWN_agg, EC_agg, Mean_agg, Mean_pred_agg, points_used_agg = [], [], [], [], [], []
            for target_date in [
                str(start_date + datetime.timedelta(days=x))
                for x in range(eval_days)
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
                        TWN_EC_split = hp_inst["TWN_EC_split"],
                        date_efficient=hp_inst["date_efficient"],
                        region_efficient=hp_inst["region_efficient"],
                        drop_columns=hp_inst['drop_columns'],
                        include_only_columns=hp_inst['include_only_columns'],
                        label=hp_inst["TWN_high"],
                    )
                    if wrangle_status == 1:
                        continue
                    points_used = train(
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
                    )
                    points_used_agg.append(points_used)
                    predict(
                        label=hp_inst["label"],
                        precision=hp_inst["precision"],
                        target_date=target_date,
                        normalize_data=hp_inst["normalize_data"],
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
                    if hp["exit_on_exception"]:
                        sys.exit(1)
            if wrangle_status == 1:
                break
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
                    "ML_ave": sum(ML_agg) / len(ML_agg),
                    "TWN_ave": sum(TWN_agg) / len(TWN_agg),
                    "EC_ave": sum(EC_agg) / len(EC_agg),
                    "Mean_ave": sum(Mean_agg) / len(Mean_agg),
                    "ML": ML_agg,
                    "TWN": TWN_agg,
                    "EC": EC_agg,
                    "mean": Mean_agg,
                    "points_used": points_used_agg,
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
    analyze()
