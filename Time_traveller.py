import logging
import requests
import sys
import os
import datetime
import time
from shutil import copyfile

from Train import train


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - "+
	"%(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"))
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)

loggr.info('Time Travellin...')


train(time_span=6,target_date='2018-10-14')

'''
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
'''