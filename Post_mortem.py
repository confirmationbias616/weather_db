import pandas as pd
import datetime
import logging
import sys
import numpy as np
import os


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line "
        + "%(lineno)d"
    )
)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def post_mortem(**kwargs):
    try:
        fc_date = kwargs["target_date"]
        if fc_date == str(datetime.datetime.now().date() - datetime.timedelta(1)):
            time_travel = False
        else:
            time_travel = True
    except KeyError:
        fc_date = str(datetime.datetime.now().date() - datetime.timedelta(1))
        time_travel = False

    loggr.info("Predicting for date: {}...".format(fc_date))

    actual_date = str(
        datetime.date(int(fc_date[:4]), int(fc_date[5:7]), int(fc_date[8:]))
        + datetime.timedelta(days=1)
    )
    if time_travel:
        time_travel_string = "time_travel/{} -> ".format(datetime.datetime.now().date())
    else:
        time_travel_string = ""
    
   
    dbh = pd.read_csv('/Users/Alex/Coding/weather_db/Data/history_db.csv')
    dbp = pd.read_csv('/Users/Alex/Coding/weather_db/Data/prediction_db.csv')

    dbp = dbp.drop('high', axis=1).merge(dbh[dbh.provider=='TWN'][['date', 'region', 'province', 'high']], how='left', on=['date', 'province', 'region']).reset_index().drop('index', axis=1)
    dbp.to_csv('/Users/Alex/Coding/weather_db/Data/prediction_db.csv', index=False)

    dbp.dropna(axis=0, subset=['high', 'predictions'], inplace=True)

    dbp['ML_real_diff'] = dbp.high-dbp.predictions
    dbp['ML_real_diff_r1'] = dbp.high-round(dbp.predictions,1)
    dbp['ML_real_diff_r0'] = dbp.high-round(dbp.predictions,0)
    dbp['mean_real_diff'] = dbp.high-dbp.mean_high_T1
    dbp['TWN_real_diff'] = dbp.high-dbp.TWN_high_T1
    dbp['EC_real_diff'] = dbp.high-dbp.EC_high_T1
    dbp['ave_real_diff'] = dbp.high-dbp.rolling_normal_high
    dbp['ML_real_diff_abs'] = abs(dbp.high-dbp.predictions)
    dbp['ML_real_diff_r1_abs'] = abs(dbp.high-round(dbp.predictions,1))
    dbp['ML_real_diff_r0_abs'] = abs(dbp.high-round(dbp.predictions,0))
    dbp['mean_real_diff_abs'] = abs(dbp.high-dbp.mean_high_T1)
    dbp['TWN_real_diff_abs'] = abs(dbp.high-dbp.TWN_high_T1)
    dbp['EC_real_diff_abs'] = abs(dbp.high-dbp.EC_high_T1)
    dbp['ave_real_diff_abs'] = abs(dbp.high-dbp.rolling_normal_high)

    dbp.to_csv('/Users/Alex/Coding/weather_db/Data/prediction_db_analysis.csv', index=False)

    for column in ['ML_real_diff', 'ML_real_diff_r1', 'ML_real_diff_r0', 'mean_real_diff', 'TWN_real_diff', 'EC_real_diff', 'ave_real_diff']:
        loggr.info('column {} as an average of {} and an rmse of {}'. format(column, sum(dbp[column].apply(abs))/len(dbp), (sum(dbp[column].apply(lambda x: x**2))/len(dbp))**0.5))

    for column in ['ML_real_diff_abs', 'ML_real_diff_r1_abs', 'ML_real_diff_r0_abs', 'mean_real_diff_abs', 'TWN_real_diff_abs', 'EC_real_diff_abs', 'ave_real_diff_abs']:
        loggr.info('column {} as an average of {} and an rmse of {}'. format(column, sum(dbp[column].apply(abs))/len(dbp), (sum(dbp[column].apply(lambda x: x**2))/len(dbp))**0.5))
