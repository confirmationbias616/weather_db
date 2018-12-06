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

    dbp['ML_real_diff'] = abs(dbp.high-dbp.predictions)
    dbp['ML_real_diff_r1'] = abs(dbp.high-round(dbp.predictions,1))
    dbp['ML_real_diff_r0'] = abs(dbp.high-round(dbp.predictions,0))
    dbp['mean_real_diff'] = abs(dbp.high-dbp.mean_high_T1)
    dbp['TWN_real_diff'] = abs(dbp.high-dbp.TWN_high_T1)
    dbp['EC_real_diff'] = abs(dbp.high-dbp.EC_high_T1)
    dbp['ave_real_diff'] = abs(dbp.high-dbp.rolling_normal_high)
    dbp['ML_win'] = (dbp['ML_real_diff']<dbp['TWN_real_diff']) & (dbp['ML_real_diff']<dbp['EC_real_diff'])
    dbp['TWN_win'] = (dbp['TWN_real_diff']<dbp['EC_real_diff']) & (dbp['TWN_real_diff']<dbp['ML_real_diff'])
    dbp['EC_win'] = (dbp['EC_real_diff']<dbp['TWN_real_diff']) & (dbp['EC_real_diff']<dbp['ML_real_diff'])

    dbp.to_csv('/Users/Alex/Coding/weather_db/Data/prediction_db_analysis.csv', index=False)

    loggr.info('For entire predictions history:')
    for column in ['ML_real_diff', 'ML_real_diff_r1', 'ML_real_diff_r0', 'mean_real_diff', 'TWN_real_diff', 'EC_real_diff', 'ave_real_diff']:
        loggr.info('\tcolumn {} has an average of {} and an rmse of {}'. format(column, sum(dbp[column])/len(dbp), (sum(dbp[column].apply(lambda x: x**2))/len(dbp))**0.5))

    dbp_latest = dbp[dbp.date==fc_date]
    loggr.info('For yesterday only:')
    for column in ['ML_real_diff', 'ML_real_diff_r1', 'ML_real_diff_r0', 'mean_real_diff', 'TWN_real_diff', 'EC_real_diff', 'ave_real_diff']:
        loggr.info('\tcolumn {} has an average of {} and an rmse of {}'. format(column, sum(dbp_latest[column])/len(dbp_latest), (sum(dbp_latest[column].apply(lambda x: x**2))/len(dbp_latest))**0.5))

    loggr.info("Total points to share: {}".format(len(dbp.ML_win)))
    loggr.info("Total ML wins: {}".format(sum(dbp.ML_win)))
    loggr.info("Total TWN wins: {}".format(sum(dbp.TWN_win)))
    loggr.info("Total EC wins: {}".format(sum(dbp.EC_win)))

    loggr.info("Latest points to share: {}".format(len(dbp_latest.ML_win)))
    loggr.info("Latest ML wins: {}".format(sum(dbp_latest.ML_win)))
    loggr.info("Latest TWN wins: {}".format(sum(dbp_latest.TWN_win)))
    loggr.info("Latest EC wins: {}".format(sum(dbp_latest.EC_win)))
