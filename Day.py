import logging
import requests
import sys
import os
import datetime
import time
from shutil import copyfile
from Train import train
from Post_mortem import post_mortem

#Hyperparameters
time_span = 10

PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line "
        + "%(lineno)d"
    )
)
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)
loggr.info("Starting ETL process...")

while True:
    try:
        copyfile(
            "{}/Data/history_db.csv".format(PATH),
            "{}/Data/history_db_TEMP.csv".format(PATH),
        )  # create temporary backup of csv file
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
        copyfile(
            "{}/Data/history_db_TEMP.csv".format(PATH),
            "{}/Data/history_db.csv".format(PATH),
        )  # revert csv file back to temporary backup to delete today's failed attempt
        time.sleep(10)
        continue
    except Exception as e:
        loggr.exception("ETL_history.py could not run. Here's why: \n {e}")
        continue
    finally:
        os.remove(
            "{}/Data/history_db_TEMP.csv".format(PATH)
        )  # clean up by deleting the temp backup
try:
    loggr.info("Running a few tests...")
    from Test_data import check_historical_data
    from Test_data import correct_bad_EC_data
    check_historical_data()
    correct_bad_EC_data()
    loggr.info("Tests are now complete.")
except Exception as e:
    loggr.exception("Test_data.py could not run. Here's why: \n {e}")
try:
    loggr.info("Wrangling data to prep for ML " "Analysis...")
    import Wrangle
    loggr.info("'master_db.csv' is now updated with the latest weather!")
    copyfile(
        "{}/Data/master_db.csv".format(PATH),
        "/Users/Alex/Dropbox (Personal)/master_db.csv",
    )
except Exception as e:
    loggr.exception("Wrangle.py could not run. Here's why: \n {e}")
loggr.info("ETL process is now complete.")
try:
    loggr.info("Crunching the numbers with ML...")
    train(time_span=time_span)
    loggr.info("ML complete. Pickled model is ready!")
except Exception as e:
    loggr.exception("ML.py could not run. Here's why: \n {e}")
try:
    loggr.info(
        "Comparing predictions with real-world results for date of "
        "{}".format(datetime.datetime.now().date() - datetime.timedelta(days=1))
    )
    post_mortem()
    loggr.info("Post_mortem results are ready!")
except FileNotFoundError:
    loggr.critical("Missing data - cannot compare performances " "for this date")
except Exception as e:
    loggr.exception("Post_mortem.py could not run. Here's why: \n {e}")
