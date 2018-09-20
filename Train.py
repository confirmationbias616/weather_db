import pandas as pd
import datetime
import numpy as np
import logging
import sys
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - "+
    "%(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"))
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def save_model(model):
    filename = '{}/Gym/pickeled_models/{}.pkl'.format(
        PATH, datetime.datetime.now().date())
    with open(filename, 'wb') as output:
        pickle.dump(model, output)


def save_features(features):
    filename = '{}/Gym/feature_list/{}.pkl'.format(
        PATH, datetime.datetime.now().date())
    with open(filename, 'wb') as output:
        pickle.dump(features, output)


# load data
db = pd.read_csv('{}/Data/master_db.csv'.format(PATH), dtype={'date': 'str'})

# check to see if there's any duplicated entries (should be empty)
db = db.drop('Unnamed: 0', axis=1).drop_duplicates()

today = datetime.datetime.now().date()

# drop useless columns and clean the whole thing up
drop_attr = [
    'date',
    'day',
    'month',
    'year',
    # 'latitude',
    # 'longitude',
    'EC_region_code',
    'TWN_region_code',
    'EC_low',
    'EC_precipitation',
    'EC_high',
    'EC_high_2ago',
    # 'TWN_high_2ago',
    'TWN_low',
    'TWN_precipitation',
    # 'TWN_day_pop_T1',
    # 'TWN_night_pop_T1',
    # 'TWN_high_T1',
    # 'TWN_low_T1',
    # 'EC_day_pop_T1',
    # 'EC_high_T1',
    # 'EC_low_T1',
    # 'EC_night_pop_T1',
    # 'TWN_day_pop_T2',
    # 'TWN_night_pop_T2',
    # 'TWN_high_T2',
    # 'TWN_low_T2',
    # 'EC_day_pop_T2',
    # 'EC_high_T2',
    # 'EC_low_T2',
    # 'EC_night_pop_T2',
    'TWN_day_pop_T3',
    'TWN_night_pop_T3',
    'TWN_high_T3',
    'TWN_low_T3',
    'EC_day_pop_T3',
    'EC_high_T3',
    'EC_low_T3',
    'EC_night_pop_T3',
    'TWN_day_pop_T4',
    'TWN_night_pop_T4',
    'TWN_high_T4',
    'TWN_low_T4',
    'EC_day_pop_T4',
    'EC_high_T4',
    'EC_low_T4',
    'EC_night_pop_T4',
    'TWN_day_pop_T5',
    'TWN_high_T5',
    'TWN_low_T5',
    'TWN_night_pop_T5',
    'EC_day_pop_T5',
    'EC_high_T5',
    'EC_low_T5',
    'EC_night_pop_T5',
    'precipitation_x',
    'precipitation_y',
    'precipitation_x.1',
    'precipitation_y.1',
    'precipitation_x.2',
    'precipitation_y.2',
    'precipitation_x.3',
    'precipitation_y.3',
    'precipitation_x.4',
    'precipitation_y.4']

db.drop(drop_attr, axis=1, inplace=True)
db.dropna(axis=0, how='any', inplace=True)

# categorical column has to be left out of the ML models (might look into
# one-hot-encoding in the future)
db.drop(['region', 'province'], axis=1, inplace=True)

# Create X as features set and Y as labeled set
label_column = 'TWN_high'
X, y = db.drop(label_column, axis=1), db[label_column]

# choose only the latest z amount of data points to analyze (to eliminate
# negative effects of data drift)
LOOKBACK_DAYS = 1000
if type(LOOKBACK_DAYS) == int:
    X = X.tail(LOOKBACK_DAYS)
    y = y.tail(LOOKBACK_DAYS)

loggr.info(
    'Amount of data points being used in ML analysis: {}'.format(len(X.index)))

# compute for baseline error when predicting tomorrow's high using only TWN T1
# prediction
baseline_rmse = np.sqrt(mean_squared_error(y, X['TWN_high_T1']))
baseline_ave_error = sum((abs(y - X['TWN_high_T1']))) / len(y)

# save attributes that are used for training ML model -> to be deployed in our
# daily prediction later in the evening
ML_attr = X.columns
save_features(ML_attr)

# normalize
pipeline = Pipeline([('std_scaler', StandardScaler())])
X = pipeline.fit_transform(X)

# instantiate list of model that we'll be using
model = RandomForestRegressor(
    bootstrap=True,
    max_depth=49,
    max_features=9,
    min_samples_leaf=5,
    min_samples_split=2,
    n_estimators=154,
    random_state=42,
)

loggr.info('Baseline RMSE: {}'.format(baseline_rmse))
loggr.info('Baseline average error: {}'.format(baseline_ave_error))

loggr.info('model: {}'.format(model))
loggr.info('features: {}'.format(ML_attr))


# split features and labels into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=42)

# train
model.fit(X_train, y_train)

feature_importances = sorted(
    zip(model.feature_importances_, ML_attr), reverse=True)
importance_table = pd.DataFrame(
    feature_importances, columns=['importance', 'feature'])
importance_table.to_csv('{}/Gym/{}_importance.csv'.format(PATH, today))

# test!
scores = cross_val_score(model, X_test, y_test,
                         scoring='neg_mean_squared_error',
                         cv=100, n_jobs=-1, verbose=0)
model_rmses = np.sqrt(-scores)
model_rmse = sum(model_rmses) / len(model_rmses)
loggr.info('Model RMSE:{}'.format(model_rmse))

save_model(model)

summary = list(zip([
    'train data points', 'test data points', 'baseline RMSE',
    'Baseline average error', 'model', 'features', 'model RMSE'],
    [len(X_train), len(y_train), baseline_rmse,
        baseline_ave_error, model, ML_attr, model_rmse]))

pd.DataFrame(summary).to_csv('{}/Predictions/{}_summary.csv'.format(PATH, today))
