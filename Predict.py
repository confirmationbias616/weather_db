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
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def predict(precision=1, **kwargs):
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

    try:
        today = kwargs["target_date"]
        time_travel = True
    except KeyError:
        today = datetime.datetime.now().date()
        time_travel = False
    loggr.info("Predicting for date: {}...".format(today))
    if time_travel:
        time_travel_string = "time_travel/{} -> ".format(datetime.datetime.now().date())
    else:
        time_travel_string = ""
    model = load_model()
    ML_attr = load_features()
    db = pd.read_csv("{}/Data/master_db.csv".format(PATH), dtype={"date": "str"})
    db = db.drop("Unnamed: 0", axis=1)
    tomorrow = str(
        datetime.date(int(today[:4]), int(today[5:7]), int(today[8:]))
        + datetime.timedelta(days=1)
    )
    tomorrow = "{}-{}-{}".format(tomorrow[:4], tomorrow[5:7], tomorrow[8:10])
    db_tomorrow = db[db["date"] == tomorrow]
    db_tomorrow = db_tomorrow[list(ML_attr) + ["region", "province"]]
    db_tomorrow.dropna(axis=1, how="all", inplace=True)
    db_tomorrow.dropna(axis=0, how="any", inplace=True)
    loggr.info(("".join(['{}\n'.format(x) for x in db_tomorrow.columns])))
    fc_ind = db_tomorrow.reset_index()["index"]
    pipeline = Pipeline([("std_scaler", StandardScaler())])
    X_today = pipeline.fit_transform(db_tomorrow.drop(["region", "province"], axis=1))
    predictions = model.predict(X_today)
    forecast_table = db_tomorrow.loc[fc_ind][
        ["region", "province", "TWN_high_T1", "EC_high_T1"]
    ]
    forecast_table["model_predictions"] = [
        round(predictions[i], precision) for i in range(len(predictions))
    ]
    forecast_table.to_csv(
        "{}/Predictions/{}{}_predict_tm_high.csv".format(
            PATH, time_travel_string, today
        )
    )
    forecast_table.describe().to_csv(
        "{}/Predictions/{}{}_describe.csv".format(PATH, time_travel_string, today)
    )
