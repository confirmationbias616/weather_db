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
start_date = "2018-10-11"
end_date = "2018-10-13"
iterations = 2
hp = {
    "time_span": [5, 10],
    "rolling_average_window": [1, 10, 30],
    "rolling_average_min_periods": [30],
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

search_results = pd.DataFrame(columns=hp)
hp_inst = {key: 0 for key in hp}

for i in range(iterations):
    for item in hp:
        hp_inst[item] = random.choice(hp[item])

    ML, TWN, EC, Mean = [], [], [], []
    for target_date in [
        str(start_date + datetime.timedelta(days=x)) for x in range(eval_days + 1)
    ]:
        wrangle(
            rolling_average_window=hp["rolling_average_window"],
            rolling_average_min_periods=hp["rolling_average_min_periods"],
        )
        train(
            time_span=hp_inst["time_span"],
            max_depth=hp_inst["max_depth"],
            max_features=hp_inst["max_features"],
            min_samples_leaf=hp_inst["min_samples_leaf"],
            min_samples_split=hp_inst["min_samples_split"],
            n_estimators=hp_inst["n_estimators"],
            target_date=hp_inst["target_date"],
            cv=hp_inst["cv"],
            precision=hp_inst["precision"],
        )
        predict(target_date=hp_inst["target_date"])
        ML[i], TWN[i], EC[i], Mean[i] = post_mortem(target_date=hp_inst["target_date"])

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
