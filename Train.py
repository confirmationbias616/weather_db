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
loggr.addHandler(log_handler)
loggr.setLevel(logging.DEBUG)


def train(
    time_span=10,
    max_depth=49,
    max_features=9,
    min_samples_leaf=5,
    min_samples_split=2,
    n_estimators=154,
    cv=100,
    edge_forecasting=1,
    normalize_data=1,
    **kwargs
):
    def get_date_object(date):
        return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))

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
        today = str(datetime.datetime.now().date())
        time_travel = False
    loggr.info("Training for date: {}...".format(today))
    if time_travel:
        time_travel_string = "time_travel/{} -> ".format(datetime.datetime.now().date())
    else:
        time_travel_string = ""

    db = pd.read_csv("{}/Data/master_db.csv".format(PATH))

    # Get indices for selecting portion of data centered on target date by a
    # width of specified time span
    def get_time_span_indices(today):
        date_jump = int(time_span / 2)
        today = get_date_object(today)
        # If dates aren't present in DataFrame, get ones that are
        potential_start_date = today - datetime.timedelta(days=date_jump)
        if (today == datetime.datetime.now().date()) or edge_forecasting:
            potential_end_date = today
        else:
            potential_end_date = today + datetime.timedelta(days=date_jump)
        available_dates = list(db.date.unique())
        available_dates.sort()
        while True:
            if str(potential_start_date) in available_dates:
                start_date = str(potential_start_date)
                loggr.info("Selected {} as start date".format(start_date))
                break
            elif (potential_start_date - get_date_object(available_dates[0])).days < 0:
                start_date = available_dates[0]
                loggr.info("Selected {} as start date".format(start_date))
                break
            else:
                loggr.info("There was a missing date({}) in place of attempted start date for time span. Retrying...".format(potential_start_date))
                potential_start_date += datetime.timedelta(1)
        while True:
            if str(potential_end_date) in available_dates:
                end_date = str(potential_end_date)
                loggr.info("Selected {} as end date".format(end_date))
                break
            elif (get_date_object(available_dates[-1]) - potential_end_date).days < 0:
                end_date = available_dates[-1]
                loggr.info("Selected {} as end date".format(end_date))
                break
            else:
                loggr.info("There was a missing date({}) in place of attempted end date for time span. Retrying...".format(potential_end_date))
                potential_end_date -= datetime.timedelta(1)

        start_date, end_date = str(start_date), str(end_date)
        start_index = db.index[db.date == start_date][0]
        end_index = db.index[db.date == end_date][-1]

        loggr.info(
            "selected start date and end date for time span: {} -> {}".format(
                start_date, end_date
            )
        )
        return start_index, end_index

    loggr.info("Selecting dates & indices for time span training surrounding date {}...".format(today))
    start_index, end_index = get_time_span_indices(today)

    attr = list(db.columns)

    try:
        label_column = kwargs["label"]
    except KeyError:
        label_column = "TWN_high"

    # Create X as features set and y as labeled set
    X, y = db.drop(['TWN_high', 'TWN_low', 'EC_high', 'EC_low', 'TWN_precipitation', 'EC_precipitation', 'region', 'province', 'date'], axis=1), db[label_column]
    X = X[(X.index > start_index) & (X.index < end_index)]
    y = y[(y.index > start_index) & (y.index < end_index)]
    points = len(X)
    loggr.info("Amount of data points being used in ML analysis: {}".format(points))
    # compute for baseline error when predicting tomorrow's high using only TWN T1
    # prediction
    try:    
        if 'TWN_high_T1' in attr:
            baseline_rmse = np.sqrt(mean_squared_error(y, X["TWN_high_T1"]))
            baseline_ave_error = sum((abs(y - X["TWN_high_T1"]))) / len(y)
        else:
            baseline_rmse = np.sqrt(mean_squared_error(y, X["TWN_high_T1_delta"] + X['rolling_normal_high']))
            baseline_ave_error = sum((abs(y - X["TWN_high_T1_delta"] + X['rolling_normal_high']))) / len(y)
    except ValueError:
        loggr.info('Baseline rmse not available for eventual ML performance comparison :(')
    # save attributes that are used for training ML model -> to be deployed in our
    # daily prediction later in the evening
    ML_attr = X.columns
    '''
    save_features(ML_attr)
    '''
    loggr.info(
        ("Features for training:\n" + "".join(["{}\n".format(x) for x in ML_attr]))
    )

    if normalize_data:
        loggr.info("Normalizing data...")
        pipeline = Pipeline([("std_scaler", StandardScaler())])
        X = pipeline.fit_transform(X)

    model = RandomForestRegressor(
        bootstrap=True,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        random_state=42,
    )
    loggr.info("Baseline RMSE: {}".format(round(baseline_rmse, 2)))
    loggr.info("Baseline average error: {}".format(round(baseline_ave_error, 2)))
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
    importance_table.to_csv("{}/Gym/{}_importance.csv".format(PATH, today), index=False)
    scores = cross_val_score(
        model,
        X_test,
        y_test,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    model_rmses = np.sqrt(-scores)
    model_rmse = sum(model_rmses) / len(model_rmses)
    loggr.info("Model RMSE:{}".format(round(model_rmse, 2)))
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
        "{}/Predictions/{}{}_summary.csv".format(PATH, time_travel_string, today), index=False
    )
    return points
