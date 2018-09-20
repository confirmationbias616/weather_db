import logging
import requests
import sys
import os
import time
from shutil import copyfile


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - "+
	"%(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"))
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)

loggr.info('Starting ETL process...')

while True:
	try:
		# create temporary backup of csv file
		copyfile('{}/Data/forecast_db.csv'.format(PATH), '{}/Data/forecast_db_TEMP.csv'.format(PATH))
		loggr.info('Extracting forecast data available today '\
			'(using ETL_forecast.py)...')
		import ETL_forecast
		break
	except requests.exceptions.ConnectionError:
		loggr.critical('There must be a Wi-Fi connection error. '\
			'Deleting ETL_forecast progress for today, '\
			'pausing for 10 seconds and '\
			'then trying again...')
		# revert csv file back to temporary backup to delete today's failed
		# attempt
		copyfile('{}/Data/forecast_db_TEMP.csv'.format(PATH), '{}/Data/forecast_db.csv'.format(PATH))
		time.sleep(10)
		continue
	except Exception as e:
		loggr.exception("ETL_forecast.py could not run. Here's why: \n {e}")
		continue
	finally:
		# clean up by deleting the temp backup
		os.remove('{}/Data/forecast_db_TEMP.csv'.format(PATH))

loggr.info('ETL process is now complete.')

try:
	loggr.info('Running a few tests...')
	from Test_data import check_forecast_data
	check_forecast_data()
	loggr.info('Tests are now complete.')
except Exception as e:
	loggr.exception("Test_data.py could not run. Here's why: \n {e}")

try:
	loggr.info('Predicting forecast using today pickled model from today...')
	import Predict
	loggr.info('ML forecast is ready!')
except Exception as e:
	loggr.exception("Predict.py could not run. Here's why: \n {e}")

