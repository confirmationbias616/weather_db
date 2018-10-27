import logging
import requests
import sys
import os
import datetime
import time
from shutil import copyfile
import random
import pandas as pd

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

loggr.info("Time Travellin...")

# MAKE THESE HYPERPARAMETERS ACCESSIBLE TO CLI
start_date = "2018-10-13"
end_date = "2018-10-17"
iterations = 20
hp = {
    "time_span": [5, 10],
    "rolling_average_window": [1, 10, 30],
    "rolling_average_min_periods": [1],
    "time_span": [5, 10, 20, 40],
    "max_depth": [49, 100],
    "max_features": [9, 12],
    "min_samples_leaf": [4, 5],
    "min_samples_split": [2],
    "n_estimators": [145, 155, 165],
    "cv": [20, 80, 100],
    "precision": [1],
}

start_date = datetime.date(
    int(start_date[:4]), int(start_date[5:7]), int(start_date[8:])
)
end_date = datetime.date(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:]))
eval_days = int(str(end_date - start_date).split(" ")[0])

try:
	search_results = pd.read_csv("/Users/Alex/Dropbox (Personal)/HPResults.csv")
except:
	search_results = pd.DataFrame(columns=hp)



for _ in range(iterations):
    hp_inst = {key: [] for key in list(hp.keys())}
    for item in list(hp_inst.keys()):
        hp_inst[item] = random.choice(hp[item])

    ML, TWN, EC, Mean = [[0] * (eval_days + 1)] * 4
    counter = 0
    for target_date in [
        str(start_date + datetime.timedelta(days=x)) for x in range(eval_days + 1)
    ]:
        wrangle(
            rolling_average_window=hp_inst["rolling_average_window"],
            rolling_average_min_periods=hp_inst["rolling_average_min_periods"],
        )
        train(
            target_date=target_date,
            time_span=hp_inst["time_span"],
            max_depth=hp_inst["max_depth"],
            max_features=hp_inst["max_features"],
            min_samples_leaf=hp_inst["min_samples_leaf"],
            min_samples_split=hp_inst["min_samples_split"],
            n_estimators=hp_inst["n_estimators"],
            cv=hp_inst["cv"],
            precision=hp_inst["precision"],
        )
        predict(precision=hp_inst["precision"], target_date=target_date)
        ML[counter], TWN[counter], EC[counter], Mean[counter] = post_mortem(target_date=target_date)
        counter +=1

    hp_inst.update(
        {
            "start_date": start_date,
            "end_date": end_date,
            "eval_days": eval_days,
            "ML_rms": sum([x ** 2 for x in ML]) / len(ML),
            "TWN_rms": sum([x ** 2 for x in TWN]) / len(TWN),
            "EC_rms": sum([x ** 2 for x in EC]) / len(EC),
            "Mean_rms": sum([x ** 2 for x in Mean]) / len(Mean),
        }
    )
    search_results = search_results.append(hp_inst, ignore_index=True)
    search_results.to_csv("/Users/Alex/Dropbox (Personal)/HPResults.csv")
