import pandas as pd
import datetime
import logging
import sys
import os


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - "+
	"%(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"))
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)

pred = pd.read_csv('{}/Predictions/{}_predict_tm_high.csv'.format(
	PATH, datetime.datetime.now().date() - datetime.timedelta(days=2)),
	dtype={'date': 'str'}).drop('Unnamed: 0', axis=1)

actual = pd.read_csv('{}/history_db.csv'.format(PATH))

actual = actual[actual['date'] == str(
	datetime.datetime.now().date() - datetime.timedelta(
		days=1))][['region', 'high', 'province']]

try:
	comp = pred.merge(actual, on=['region', 'province'], how='left')
except KeyError:
	comp = pred.merge(actual, on=['region'], how='left')

comp['diff'] = (comp['TWN_high_T1']) - comp['model_predictions']
comp['diff_both'] = (
	comp['TWN_high_T1'] + comp['EC_high_T1']) / 2 - comp['model_predictions']
comp['diff_real'] = (comp['high']) - comp['model_predictions']
comp['diff_TWN_rival'] = (comp['high']) - comp['TWN_high_T1']
comp['diff_EC_rival'] = (comp['high']) - comp['EC_high_T1']

comp = comp.sort_values('diff_real')

ML_perf = ((comp['diff_real'].apply(
	lambda x: x ** 2).sum()) / len(comp.index)) ** 0.5

TWN_rival_perf = ((comp['diff_TWN_rival'].apply(
	lambda x: x ** 2).sum()) / len(comp.index)) ** 0.5
EC_rival_perf = ((comp['diff_EC_rival'].apply(
	lambda x: x ** 2).sum()) / len(comp.index)) ** 0.5

comp.to_csv('{}/Predictions/{}_compare_actual.csv'.format(
	PATH, datetime.datetime.now().date() - datetime.timedelta(days=1)))

loggr.info('ML performance: {}'.format(ML_perf))
loggr.info('TWN Rival performance: {}'.format(TWN_rival_perf))
loggr.info('EC Rival performance: {}'.format(EC_rival_perf))

f = open('{}/Predictions/{}_compare_actual.txt'.format(
	PATH, datetime.datetime.now().date() - datetime.timedelta(days=1)), "w")

f.write('ML performance: {}\n'.format(ML_perf))
f.write('TWN Rival performance: {}'.format(TWN_rival_perf))
f.write('EC Rival performance: {}'.format(EC_rival_perf))
