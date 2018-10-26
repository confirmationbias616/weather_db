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
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line "
        + "%(lineno)d"
    )
)
log_handler.setLevel(logging.INFO)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)


def train(time_span=10, **kwargs):
    def save_model(model):
        filename = "{}/Gym/pickeled_models/{}{}.pkl".format(
            PATH, time_travel_string, today
        )
        with open(filename, "wb") as output:
            pickle.dump(model, output)

    def save_features(features):

        filename = "{}/Gym/feature_list/{}{}.pkl".format(
            PATH, time_travel_string, today
        )
        with open(filename, "wb") as output:
            pickle.dump(features, output)

    try:
        today = kwargs["target_date"]
        time_travel = True
    except KeyError:
        today = datetime.datetime.now().date()
        time_travel = False
    loggr.info("Training for date: {}...".format(today))
    if time_travel:
        time_travel_string = "time_travel/{} -> ".format(datetime.datetime.now().date())
    else:
        time_travel_string = ""

    db = pd.read_csv("{}/Data/master_db.csv".format(PATH), dtype={"date": "str"})
    db = db.drop("Unnamed: 0", axis=1).drop_duplicates()

    # Get indices for selecting portion of data centered on target date by a
    # width of specified time span
    def get_time_span_indices(today):
        date_jump = int(time_span / 2)
        today = datetime.date(int(today[:4]), int(today[5:7]), int(today[8:]))
        # If dates aren't present in DataFrame, get ones that are
        while True:
            try:
                start_date = today - datetime.timedelta(days=date_jump)
                # If running normally (from Day.py), time span can only go into past - as we are
                # at the edge of our records. Therefore, need to set 
                if today == datetime.datetime.now().date():
                    end_date = today
                else:
                    end_date = today + datetime.timedelta(days=date_jump)
                start_date, end_date = str(start_date), str(end_date)
                start_index = db.index[db.date == start_date][0]
                end_index = db.index[db.date == end_date][-1]
                break
            except IndexError:
                date_jump -= 1
        loggr.info(
            "selected start date and end date for time span: {} -> {}".format(
                start_date, end_date
            )
        )
        return start_index, end_index

    start_index, end_index = get_time_span_indices(today)
    # drop useless columns and clean the whole thing up
    drop_attr = [
        "date",
        "day",
        "month",
        "year",
        # 'latitude',
        # 'longitude',
        "EC_region_code",
        "TWN_region_code",
        "EC_low",
        "EC_precipitation",
        "EC_high",
        "EC_high_2ago",
        # 'TWN_high_2ago',
        "TWN_low",
        "TWN_precipitation",
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
        "TWN_day_pop_T3",
        "TWN_night_pop_T3",
        "TWN_high_T3",
        "TWN_low_T3",
        "EC_day_pop_T3",
        "EC_high_T3",
        "EC_low_T3",
        "EC_night_pop_T3",
        "TWN_day_pop_T4",
        "TWN_night_pop_T4",
        "TWN_high_T4",
        "TWN_low_T4",
        "EC_day_pop_T4",
        "EC_high_T4",
        "EC_low_T4",
        "EC_night_pop_T4",
        "TWN_day_pop_T5",
        "TWN_high_T5",
        "TWN_low_T5",
        "TWN_night_pop_T5",
        "EC_day_pop_T5",
        "EC_high_T5",
        "EC_low_T5",
        "EC_night_pop_T5",
        "precipitation_x",
        "precipitation_y",
        "precipitation_x.1",
        "precipitation_y.1",
        "precipitation_x.2",
        "precipitation_y.2",
        "precipitation_x.3",
        "precipitation_y.3",
        "precipitation_x.4",
        "precipitation_y.4",
    ]
    db.drop(drop_attr, axis=1, inplace=True)
    db.dropna(axis=0, how="any", inplace=True)
    # categorical column has to be left out of the ML models (might look into
    # one-hot-encoding in the future)
    db.drop(["region", "province"], axis=1, inplace=True)
    # Create X as features set and Y as labeled set
    label_column = "TWN_high"
    X, y = db.drop(label_column, axis=1), db[label_column]
    X = X[(X.index > start_index) & (X.index < end_index)]
    y = y[(y.index > start_index) & (y.index < end_index)]
    loggr.info(
        "Amount of data points being used in ML analysis: {}".format(len(X.index))
    )
    # compute for baseline error when predicting tomorrow's high using only TWN T1
    # prediction
    baseline_rmse = np.sqrt(mean_squared_error(y, X["TWN_high_T1"]))
    baseline_ave_error = sum((abs(y - X["TWN_high_T1"]))) / len(y)
    # save attributes that are used for training ML model -> to be deployed in our
    # daily prediction later in the evening
    ML_attr = X.columns
    save_features(ML_attr)
    pipeline = Pipeline([("std_scaler", StandardScaler())])
    X = pipeline.fit_transform(X)
    model = RandomForestRegressor(
        bootstrap=True,
        max_depth=49,
        max_features=9,
        min_samples_leaf=5,
        min_samples_split=2,
        n_estimators=154,
        random_state=42,
    )
    loggr.info("Baseline RMSE: {}".format(baseline_rmse))
    loggr.info("Baseline average error: {}".format(baseline_ave_error))
    loggr.debug("model: {}".format(model))
    loggr.debug("features: {}".format(ML_attr))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, random_state=42
    )
    model.fit(X_train, y_train)
    feature_importances = sorted(zip(model.feature_importances_, ML_attr), reverse=True)
    importance_table = pd.DataFrame(
        feature_importances, columns=["importance", "feature"]
    )
    importance_table.to_csv("{}/Gym/{}_importance.csv".format(PATH, today))
    scores = cross_val_score(
        model,
        X_test,
        y_test,
        scoring="neg_mean_squared_error",
        cv=100,
        n_jobs=-1,
        verbose=0,
    )
    model_rmses = np.sqrt(-scores)
    model_rmse = sum(model_rmses) / len(model_rmses)
    loggr.info("Model RMSE:{}".format(model_rmse))
    save_model(model)
    summary = list(
        zip(
            [
                "train data points",
                "test data points",
                "baseline RMSE",
                "Baseline average error",
                "model",
                "features",
                "model RMSE",
            ],
            [
                len(X_train),
                len(y_train),
                baseline_rmse,
                baseline_ave_error,
                model,
                ML_attr,
                model_rmse,
            ],
        )
    )
    pd.DataFrame(summary).to_csv(
        "{}/Predictions/{}{}_summary.csv".format(PATH, time_travel_string, today)
    )
