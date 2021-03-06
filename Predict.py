import pandas as pd
import datetime
import logging
import sys
import os
import pickle
import numpy as np
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
    dbp = pd.read_csv("{}/Data/prediction_db.csv".format(PATH))
    dbh = pd.read_csv("{}/Data/history_db.csv".format(PATH)).drop("time", axis=1)
    X = dbpp.drop(["province", "region", "date"], axis=1)
    try:
        X.drop("predictions", axis=1, inplace=True)
        loggr.info(
            "predictions already existed so they were dropped to make room for new ones"
        )
    except KeyError:
        pass
    loggr.info(
        (
            "Features for prediction:\n"
            + "".join(["{}\n".format(feature) for feature in X.columns])
        )
    )
    X = X.reindex(
        columns=(
            ["TWN_high_T1"]
            + ["EC_high_T1"]
            + list([a for a in X.columns if a not in ["TWN_high_T1", "EC_high_T1"]])
        )
    )

    if normalize_data:
        pipeline = Pipeline([("std_scaler", StandardScaler())])
        X = pipeline.fit_transform(X)

    predictions = model.predict(X)
    dbpp["predictions"] = model.predict(X)
    dbpp.to_csv("{}/Data/prediction_prep_db.csv".format(PATH), index=False)
    tomorrow = get_date_object(today) + datetime.timedelta(1)
    dbp = dbp[
        dbp.date != str(tomorrow)
    ]  # delete tomorrow's prediction to make room for new ones
    dbp = dbp.append(dbpp, ignore_index=True)
    dbp.to_csv("{}/Data/prediction_db.csv".format(PATH), index=False)
