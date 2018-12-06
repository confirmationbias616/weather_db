import pandas as pd
import datetime
import sys
import os
import logging
import numpy as np


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line " +
        "%(lineno)d"
    )
)
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


fc, ht, rc = [
    pd.read_csv("{}/Data/{}.csv".format(PATH, x), dtype={"date": "str"})
    for x in ["forecast_db", "history_db", "region_codes"]
]

# create a copy of what was collected today
today = str(datetime.datetime.now().date())
yesterday = str(datetime.datetime.now().date() - datetime.timedelta(days=1))
today = "{}-{}-{}".format(today[:4], today[5:7], today[8:10])
yesterday = "{}-{}-{}".format(yesterday[:4], yesterday[5:7], yesterday[8:10])
fc_today = fc[fc["date"] == today]
ht_today = ht[ht["date"] == yesterday]
region_count = rc["region"].count()


# check for missed forecast data collection
def check_forecast_data():
    for provider in ["EC", "TWN"]:
        points_collected = fc_today[fc_today["provider"] == provider]["high"].count()
        if points_collected != 0:
            if points_collected / 5 != region_count:
                loggr.warning(
                    "some of {} forecast data '\
                    'was missed".format(
                        provider
                    )
                )
        else:
            loggr.critical("all {} forecast data was missed".format(provider))


# This is to fix an EC bug where it seems like they record a high of '0' when
# it should actually be NaN.
def correct_bad_EC_data():
    drop_ind = list(
        fc[(fc.high == 0) & (fc.provider == "EC")].index
    )
    if drop_ind:
        fc.loc[drop_ind, "high"] = np.nan
        fc.to_csv("{}/Data/forecast_db.csv".format(PATH))
        loggr.info("Deleted {} potentially bad entries from EC.".format(len(drop_ind)))


# check for missed historical data collection
def check_historical_data():
    for provider in ["EC", "TWN"]:
        points_collected = ht_today[ht_today["provider"] == provider]["high"].count()
        if points_collected != 0:
            if points_collected / region_count < 0.6:
                loggr.warning(
                    "a suspiciously large amount of {} historical data '\
                    'was missed".format(
                        provider
                    )
                )
        else:
            loggr.critical("all {} historical data was missed".format(provider))
