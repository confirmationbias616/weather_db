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
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line "
        + "%(lineno)d"
    )
)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def predict(precision=1, normalize_data=1, **kwargs):
    def load_model():
        filename = "{}/Gym/pickeled_models/{}{}.pkl".format(
            PATH, time_travel_string, today
        )
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)
    
    def load_features():
        filename = "{}/Gym/feature_list/{}{}.pkl".format(
            PATH, time_travel_string, today
        )
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)

    def get_date_object(date):
        return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))

    try:
        today = kwargs["target_date"]
        time_travel = True
    except KeyError:
        today = datetime.datetime.now().date()
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

    db = pd.read_csv("{}/Data/master_db.csv".format(PATH))

    try:
        attr = load_features()
    except:
        attr = db.drop(['TWN_high', 'TWN_low', 'EC_high', 'EC_low', 'date'], axis=1)

    db["year"] = db.date.apply(lambda x: get_date_object(x).year)
    db["month"] = db.date.apply(lambda x: get_date_object(x).month)
    db["day"] = db.date.apply(lambda x: get_date_object(x).day)
    tomorrow = get_date_object(today) + datetime.timedelta(days=1)
    db_tomorrow = db[(db.year == tomorrow.year) & (db.month == tomorrow.month) & (db.day == tomorrow.day)]
    db_tomorrow.dropna(axis=1, how="all", inplace=True)
    db_tomorrow.dropna(axis=0, how="any", inplace=True)
    attr = [feature for feature in attr if feature in list(db_tomorrow.columns)]
    db_tomorrow = db_tomorrow[list(attr)]
    loggr.info(
        (
            "Features for prediction:\n"
            + "".join(["{}\n".format(x) for x in db_tomorrow.columns])
        )
    )
    fc_ind = db_tomorrow.reset_index()["index"]

    X_today = db_tomorrow.drop(["region", "province"], axis=1)

    if normalize_data:
        pipeline = Pipeline([("std_scaler", StandardScaler())])
        X_today = pipeline.fit_transform(X_today)
    
    predictions = model.predict(X_today)

    class MeanRegressor(BaseEstimator, RegressorMixin):  
        """Just compares forecasts from providers and picks a number in between"""
        
        def predict(self, X, y=None):
            return (X.TWN_high_T1 + X.EC_high_T1) / 2

    mean_predictor = MeanRegressor()
    mean_predictions = mean_predictor.predict(X_today)

    try:
        _ = db_tomorrow["TWN_high_T1"]
        _ = db_tomorrow["EC_high_T1"]
    except KeyError:
        db_tomorrow["TWN_high_T1"] = (
            db_tomorrow["TWN_high_T1_delta"] + db_tomorrow["rolling_normal_high"]
        )
        db_tomorrow["EC_high_T1"] = (
            db_tomorrow["EC_high_T1_delta"] + db_tomorrow["rolling_normal_high"]
        )

    forecast_table = db_tomorrow.loc[fc_ind][
        ["region", "province", "TWN_high_T1", "EC_high_T1"]
    ]
    forecast_table["model_predictions"] = [
        round(predictions[i], precision) for i in range(len(predictions))
    ]
    forecast_table["mean_predictions"] = [
        round(mean_predictions[i], precision) for i in range(len(mean_predictions))
    ]
    forecast_table.to_csv(
        "{}/Predictions/{}{}_predict_tm_high.csv".format(
            PATH, time_travel_string, today
        ), index=False
    )
    forecast_table.describe().to_csv(
        "{}/Predictions/{}{}_describe.csv".format(PATH, time_travel_string, today), index=False
    )
