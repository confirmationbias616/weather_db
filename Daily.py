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

from Test_data import check_historical_data
from Test_data import correct_bad_EC_data
from Test_data import check_forecast_data
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
        return {}

def get_datetime(date):
    return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))

hp = load_hyperparameters()
hp.update({
    "iterations": 1,
    "start_date": str(datetime.datetime.now().date()),
    "end_date": str(datetime.datetime.now().date()),
    "edge_forecasting": 1,
    "real_time": 1,
    "exit_on_exception": 1
})

loggr.info("Starting ETL process...")

while True:
    try:
        loggr.info("Extracting Yesterday's recorded data " "(using ETL_history.py)...")
        import ETL_history
        break
    except requests.exceptions.ConnectionError:
        loggr.critical(
            "There must be a Wi-Fi connection error. "
            "Deleting ETL_history progress for today, "
            "pausing for 10 seconds and "
            "then trying again..."
        )
        time.sleep(10)
        continue
    except Exception as e:
        loggr.exception("ETL_history.py could not run. Here's why: \n {e}")
        continue

try:
    loggr.info("Extracting current weather conditions (using ETL_current.py)...")
    import ETL_current
    loggr.info("ETL current process is now complete.")
except Exception as e:
    loggr.exception("ETL_current.py could not run. Here's why: \n {e}")

try:
    loggr.info("Running a few tests...")
    check_historical_data()
    correct_bad_EC_data()
    loggr.info("Tests are now complete.")
except Exception as e:
    loggr.exception("Test_data.py could not run. Here's why: \n {e}")

while True:
    try:
        loggr.info("Extracting forecast data available today " "(using ETL_forecast.py)...")
        import ETL_forecast
        break
    except requests.exceptions.ConnectionError:
        loggr.critical(
            "There must be a Wi-Fi connection error. "
            "Deleting ETL_forecast progress for today, "
            "pausing for 10 seconds and "
            "then trying again..."
        )
        time.sleep(10)
        continue
    except Exception as e:
        loggr.exception("ETL_forecast.py could not run. Here's why: \n {e}")
        continue
loggr.info("ETL forecast process is now complete. Pausing 1 minute to make sure csv has time to save propperly.")
time.sleep(60)

try:
    loggr.info("Running a few tests...")
    check_forecast_data()
    loggr.info("Tests are now complete.")
except Exception as e:
    loggr.exception("Test_data.py could not run. Here's why: \n {e}")

###Post_mortem (Dates will be tricky here!)
try:
    loggr.info("Preparing predictions table for tomorrow ({})".format(datetime.datetime.now().date() + datetime.timedelta(1)))
    post_mortem(target_date=str(get_datetime(hp['start_date'])-datetime.timedelta(1)))
    loggr.info("Post_mortem results for yesterday ({}) are now ready".format(datetime.datetime.now().date() - datetime.timedelta(1)))
except Exception as e:
    loggr.exception("Post_mortem.py could not run. Here's why: \n {e}")

try:
    loggr.info("Wrangling data before training.")
    wrangle_status = wrangle(
        target_date=hp['start_date'],
        time_span=hp["time_span"],
        rolling_average_window=hp["rolling_average_window"],
        rolling_average_min_periods=hp[
            "rolling_average_min_periods"
        ],
        TWN_EC_split = hp["TWN_EC_split"],
        date_efficient=hp["date_efficient"],
        region_efficient=hp["region_efficient"],
        drop_columns=hp['drop_columns'],
        include_only_columns=hp['include_only_columns'],
        label=hp["label"],
        real_time=hp["real_time"],
        )
    loggr.info("Wrangling Complete")
except Exception as e:
    loggr.exception("Wrangle.py could not run. Here's why: \n {e}")

try:
    loggr.info("Training on wrangled data to produce a model for evening predictions".format(datetime.datetime.now().date() + datetime.timedelta(1)))
    train(
        target_date=hp['start_date'],
        label=hp["label"],
        time_span=hp["time_span"],
        max_depth=hp["max_depth"],
        max_features=hp["max_features"],
        min_samples_leaf=hp["min_samples_leaf"],
        min_samples_split=hp["min_samples_split"],
        n_estimators=hp["n_estimators"],
        cv=hp["cv"],
        precision=hp["precision"],
        edge_forecasting=hp["edge_forecasting"],
        normalize_data=hp["normalize_data"],
    )
    loggr.info("ML Model is now ready")
except Exception as e:
    loggr.exception("Train.py could not run. Here's why: \n {e}")

try:
    loggr.info("Preparing predictions table for tomorrow ({})".format(datetime.datetime.now().date()+datetime.timedelta(1)))
    wrangle_status = wrangle(
        target_date=hp['start_date'],
        time_span=hp["time_span"],
        rolling_average_window=hp["rolling_average_window"],
        rolling_average_min_periods=hp[
            "rolling_average_min_periods"
        ],
        TWN_EC_split = hp["TWN_EC_split"],
        date_efficient=hp["date_efficient"],
        region_efficient=hp["region_efficient"],
        drop_columns=hp['drop_columns'],
        include_only_columns=hp['include_only_columns'],
        label=hp["label"],
        real_time=hp["real_time"],
        )
    loggr.info("Predictions table is now ready")
except Exception as e:
    loggr.exception("Wrangle.py could not run. Here's why: \n {e}")

try:
    loggr.info("Running predictions for tomorrow ({})".format(datetime.datetime.now().date() + datetime.timedelta(1)))
    predict(
        label=hp["label"],
        precision=hp["precision"],
        target_date=hp['start_date'],
        normalize_data=hp["normalize_data"],
    )

    loggr.info("Predictions are now ready")
except Exception as e:
    loggr.exception("Predict.py could not run. Here's why: \n {e}")

loggr.info('All done for today!')
