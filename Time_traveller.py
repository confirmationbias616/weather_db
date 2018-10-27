import logging
import requests
import sys
import os
import datetime
import time
from shutil import copyfile

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
time_span = 10
rolling_average_window = 10
rolling_average_min_periods = 1


start_date = datetime.date(
    int(start_date[:4]), int(start_date[5:7]), int(start_date[8:])
)
end_date = datetime.date(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:]))
day_span = int(str(end_date - start_date).split(" ")[0])

for target_date in [
    str(start_date + datetime.timedelta(days=x)) for x in range(1, day_span + 1)
]:
    wrangle(
        rolling_average_window=rolling_average_window,
        rolling_average_min_periods=rolling_average_min_periods,
    )
    train(time_span=time_span, target_date=target_date)
    predict(target_date=target_date)
    post_mortem(target_date=target_date)
