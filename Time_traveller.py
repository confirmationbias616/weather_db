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
target_date = "2018-10-15"
time_span = 10
rolling_average_window = 10
rolling_average_min_periods = 1

wrangle(
    rolling_average_window=rolling_average_window,
    rolling_average_min_periods=rolling_average_min_periods,
)
train(time_span=time_span, target_date=target_date)
predict(target_date=target_date)
post_mortem(target_date=target_date)


"""
try:
	loggr.info('Crunching the numbers with ML...')
	import Train
	loggr.info('ML complete. Pickled model is ready!')
except Exception as e:
	loggr.exception("ML.py could not run. Here's why: \n {e}")

try:
	loggr.info('Comparing predictions with real-world results for date of '\
		'{}'.format(datetime.datetime.now().date() - datetime.timedelta(days=1)))
	import Post_mortem
	loggr.info('Post_mortem results are ready!')
except FileNotFoundError:
		loggr.critical('Missing data - cannot compare performances '\
			'for this date')
except Exception as e:
	loggr.exception("Post_mortem.py could not run. Here's why: \n {e}")
"""
