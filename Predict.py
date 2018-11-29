import pandas as pd
import datetime
import logging
import sys
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import model_selection
from sklearn.ensemble import BaggingRegressor


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
loggr.setLevel(logging.DEBUG)


def predict(precision=1, normalize_data=1, **kwargs):
    def load_model():
        filename = "{}/Gym/pickeled_models/{}{}.pkl".format(
            PATH, time_travel_string, today
        )
        loggr.debug("Loading model {}".format(filename))
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)
    
    def load_features():
        filename = "{}/Gym/feature_list/{}{}.pkl".format(
            PATH, time_travel_string, today
        )
        loggr.debug("Loading features {}".format(filename))
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)

    def get_date_object(date):
        return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))

    try:
        today = kwargs["target_date"]
        if today == str(datetime.datetime.now().date()):
            time_travel = False
        else:
            time_travel = True
    except KeyError:
        today = str(datetime.datetime.now().date())
        time_travel = False

    try:
        label_column = kwargs["label"]
    except KeyError:
        label_column = "TWN_high"

    loggr.info("Predicting for date: {}...".format(today))
    if time_travel:
        time_travel_string = "time_travel/{} -> ".format(datetime.datetime.now().date())
    else:
        time_travel_string = ""

    model = load_model()
    dbpp = pd.read_csv("{}/Data/prediction_prep_db.csv".format(PATH))
    dbh = pd.read_csv("{}/Data/history_db.csv".format(PATH)).drop("time", axis=1)
    X = dbpp.drop(['province', 'region', 'date'], axis=1)
    loggr.info(("Features for prediction:\n" + "".join(["{}\n".format(feature) for feature in X.columns])))
    X = X.reindex(columns=(['TWN_high_T1'] + ['EC_high_T1'] + list([a for a in X.columns if a not in ['TWN_high_T1', 'EC_high_T1']])))

    if normalize_data:
        pipeline = Pipeline([("std_scaler", StandardScaler())])
        X = pipeline.fit_transform(X)

    predictions = model.predict(X)
    dbpp['predictions'] = model.predict(X)
    dbpp.to_csv("{}/Data/prediction_prep_db.csv".format(PATH))

    dbpp['date'] = today
    dbh = dbh[dbh.provider=='TWN'][['date', 'region', 'province', 'high']]
    dbppu = dbpp.merge(dbh, on=['date', 'region', 'province'])
    dbppu.to_csv("{}/Data/prediction_prep_db_UPDATED.csv".format(PATH), index=False)

    '''
    class MeanRegressor(BaseEstimator, RegressorMixin):  
        """Just compares forecasts from providers and picks a number in between"""
        
        def predict(self, X, y=None):
            loggr.debug("dealing with data of type: {}".format(type(X)))
            try:
                return (X[:,0] + X[:,1]) / 2
            except TypeError:
                return (X['TWN_high_T1'] + X['EC_high_T1']) / 2

    mean_predictor = MeanRegressor()
    mean_predictions = list(mean_predictor.predict(X))


    kfold = model_selection.KFold(n_splits=10, random_state=42)
    # create the sub models
    estimators = []
    model1 = model
    estimators.append(('MLRF', model1))
    model2 = MeanRegressor()
    estimators.append(('Mean', model2))
    # create the ensemble model
    ensemble = BaggingRegressor(estimators)
    results = model_selection.cross_val_score(ensemble, X_today, g_truth, cv=kfold)
    '''

    try:
        _ = dbpp["TWN_high_T1"]
        _ = dbpp["EC_high_T1"]
    except KeyError:
        dbpp["TWN_high_T1"] = (
            dbpp["TWN_high_T1_delta"] + dbpp["rolling_normal_high"]
        )
        dbpp["EC_high_T1"] = (
            dbpp["EC_high_T1_delta"] + dbpp["rolling_normal_high"]
        )

    forecast_table = dbpp[
        ["region", "province", "TWN_high_T1", "EC_high_T1"]
    ]
    forecast_table["model_predictions"] = [
        round(predictions[i], precision) for i in range(len(predictions))
    ]

    forecast_table.to_csv(
        "{}/Predictions/{}{}_predict_tm_high.csv".format(
            PATH, time_travel_string, today
        ), index=False
    )
    forecast_table.describe().to_csv(
        "{}/Predictions/{}{}_describe.csv".format(PATH, time_travel_string, today), index=False
    )

