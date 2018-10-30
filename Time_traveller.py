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
log_handler.setLevel(logging.INFO)
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
            "edge_forecasting": [1, 0],
            "rolling_average_window": [30],
            "rolling_average_min_periods": [1],
            "max_depth": [100],
            "max_features": [12],
            "min_samples_leaf": [5],
            "min_samples_split": [2],
            "n_estimators": [155],
            "cv": [100],
            "precision": [1],
        }


hp = load_hyperparameters()

start_date = datetime.date(
    int(hp["start_date"][:4]), int(hp["start_date"][5:7]), int(hp["start_date"][8:])
)
end_date = datetime.date(
    int(hp["end_date"][:4]), int(hp["end_date"][5:7]), int(hp["end_date"][8:])
)
try:
    eval_days = int(str(end_date - start_date).split(" ")[0])
except ValueError:
    eval_days = 0

try:
    search_results = pd.read_csv("/Users/Alex/Dropbox (Personal)/HPResults.csv")
except FileNotFoundError:
    search_results = pd.DataFrame(columns=(["log_time"] + list(hp.keys())))

loggr.info("Time Travellin...")

for i in range(hp['iterations']):
    while True:
        try:
            hp = load_hyperparameters()
            hp_inst = {key: [] for key in list(hp.keys())}
            for item in list(hp_inst.keys()):
                if type(hp[item]) is list:
                    hp_inst[item] = random.choice(hp[item])
                else:
                    hp_inst[item] = hp[item]
            loggr.info(
                "hperparameters randomly selected for this loop:\n"
                + "".join(["{}:{}\n".format(x, hp_inst[x]) for x in hp_inst])
            )

            ML_agg, TWN_agg, EC_agg, Mean_agg, points_used_agg = [], [], [], [], []
            for target_date in [
                str(start_date + datetime.timedelta(days=x))
                for x in range(eval_days + 1)
            ]:
                loggr.info(
                    "Testing random hyperparameter set {} of {} on date {}".format(
                        i + 1, hp['iterations'], target_date
                    )
                )
                wrangle(
                    rolling_average_window=hp_inst["rolling_average_window"],
                    rolling_average_min_periods=hp_inst["rolling_average_min_periods"],
                )
                points_used = train(
                    target_date=target_date,
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
                predict(precision=hp_inst["precision"], target_date=target_date)
                ML, TWN, EC, Mean = post_mortem(target_date=target_date)
                ML_agg.append(ML)
                TWN_agg.append(TWN)
                EC_agg.append(EC)
                Mean_agg.append(Mean)

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
            search_results = search_results.append(hp_inst, ignore_index=True)

            try:
                search_results.to_csv(
                    "/Users/Alex/Dropbox (Personal)/HPResults.csv", index=False
                )
            except FileNotFoundError:
                pass
        except Exception as e:
            loggr.exception("this loop could not finish. Here's why: \n {e}")
            loggr.exception("Abandoning this loop and skipping to the next one...")
            continue
        break
