import pandas as pd
import datetime
import logging
import sys
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - "+
    "%(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"))
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def load_model():
    filename = '{}/Gym/pickeled_models/{}.pkl'.format(
        PATH, datetime.datetime.now().date())
    with open(filename, 'rb') as input_file:
        pickle.load(input_file)


def load_features():
    filename = '{}/Gym/feature_list/{}.pkl'.format(
        PATH, datetime.datetime.now().date())
    with open(filename, 'rb') as input_file:
        pickle.load(input_file)


model = load_model()

ML_attr = load_features()

# load data
db = pd.read_csv('{}/Data/master_db.csv'.format(PATH), dtype={'date': 'str'})

# check to see if there's any duplicated entries (should be empty)
db = db.drop('Unnamed: 0', axis=1).drop_duplicates()

today = datetime.datetime.now().date()
tomorrow = str(datetime.datetime.now().date() + datetime.timedelta(days=1))

# use the line below if time traveling in the past
# tomorrow = tomorrow-datetime.timedelta(days=2))

# just some formatting
tomorrow = '{}-{}-{}'.format(tomorrow[:4], tomorrow[5:7], tomorrow[8:10])

db_tomorrow = db[db['date'] == tomorrow]

# drop all columns that are not being used at all to train the model
db_tomorrow = db_tomorrow[ML_attr]

# drop any column that is completely empty
db_tomorrow.dropna(axis=1, how='all', inplace=True)

# drop any row that is completely empty
db_tomorrow.dropna(axis=0, how='any', inplace=True)

fc_ind = db_tomorrow.reset_index()['index']

# normalize
pipeline = Pipeline([('std_scaler', StandardScaler())])
X_today = pipeline.fit_transform(db_tomorrow)

predictions = model.predict(X_today)

forecast_table = db_tomorrow.loc[
    fc_ind][['region', 'province', 'TWN_high_T1', 'EC_high_T1']]


forecast_table['model_predictions'] = [
    round(predictions[i]) for i in range(len(predictions))]

forecast_table.to_csv('{}/Predictions/{}_predict_tm_high.csv'.format(PATH, today))
forecast_table.describe().to_csv('{}/Predictions/{}_describe.csv'.format(PATH, today))
