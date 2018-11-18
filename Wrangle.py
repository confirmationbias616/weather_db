import logging
import sys
import pandas as pd
import datetime
import os
import numpy as np


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"
    )
)
loggr.addHandler(log_handler)
loggr.setLevel(logging.DEBUG)


def wrangle(
    time_span=10,
    rolling_average_window=30,
    rolling_average_min_periods=1,
    date_efficient=True,
    region_efficient=False,
    **kwargs
):
    def shrink_dates(df):
        pillow = time_span + 10 + rolling_average_window
        today_date = get_date_object(today)
        return df[
            (df["date"] < str(today_date + datetime.timedelta(days=pillow)))
            & (df["date"] > str(today_date - datetime.timedelta(days=pillow)))
        ]

    def shrink_regions(df):
        return df[
            ((df["longitude"] * -0.09) + 43 > df["latitude"]) & (df["latitude"] < 53)
        ]

    def get_date_object(date):
        return datetime.date(int(date[:4]), int(date[5:7]), int(date[8:]))

    loggr.info("Starting to Wrangle data into fresh version of master_db.csv...")
    try:
        today = kwargs["target_date"]
    except KeyError:
        today = str(datetime.datetime.now().date())

    loggr.info("Loading all the relevant data")
    try:
        dbh = pd.read_csv("{}/Data/history_db.csv_TEMP".format(PATH)).drop(
            "time", axis=1
        )
    except FileNotFoundError:
        dbh = pd.read_csv("{}/Data/history_db.csv".format(PATH)).drop("time", axis=1)
    try:
        dbf = pd.read_csv("{}/Data/forecast_db_TEMP.csv".format(PATH)).drop(
            "time", axis=1
        )
    except FileNotFoundError:
        dbf = pd.read_csv("{}/Data/forecast_db.csv".format(PATH)).drop("time", axis=1)
    dbc = pd.read_csv("{}/Data/current_db.csv".format(PATH))
    dba = pd.read_csv("{}/Data/dba.csv".format(PATH))
    dbll = pd.read_csv(
        "{}/Data/region_codes.csv".format(PATH),
        dtype={"latitude": "float", "longitude": "float"},
    )

    [dbh, dbf, dbc, dba, dbll] = [
        df[df.columns.drop(list(df.filter(regex="Unnamed")))]
        for df in [dbh, dbf, dbc, dba, dbll]
    ]

    if date_efficient:
        loggr.info("Shrinking data for speed...")
        loggr.debug("Initially, df size was dbh:{}, dbf:{}, dbc:{}, dba:{}".format(len(dbh), len(dbf), len(dbc), len(dba)))
        [dbh, dbf, dbc, dba] = [shrink_dates(df) for df in [dbh, dbf, dbc, dba]]

    fc_providers = dbf.provider.unique()
    fc_days = dbf.day.unique()
    seg_dbf_list = [0] * len(fc_providers) * len(fc_days)
    seg_num = 0
    for provider in fc_providers:
        for day in fc_days:
            seg_dbf_list[seg_num] = dbf[
                (dbf.day == day) & (dbf.provider == provider)
            ].drop(["day", "provider"], axis=1)
            columns_to_reanme = list(dbf.columns)
            for column in ["date", "province", "region"]:
                columns_to_reanme.remove(column)
            seg_dbf_list[seg_num].rename(
                columns={
                    x: "{}_{}_T{}".format(provider, x, day) for x in columns_to_reanme
                },
                inplace=True,
            )
            seg_dbf_list[seg_num].date = seg_dbf_list[seg_num].date.apply(lambda x: str(get_date_object(x) + datetime.timedelta(int(day))))
            seg_num += 1

    i = 0
    for seg_dbf in seg_dbf_list:
        if i == 0:
            db = seg_dbf
        else:
            db = db.merge(seg_dbf_list[i], on=["date", "region", "province"])
        i += 1

    h_providers = dbh.provider.unique()
    seg_dbh_list = [0] * len(h_providers)
    seg_num = 0
    for provider in h_providers:
        seg_dbh_list[seg_num] = dbh[dbh.provider == provider].drop("provider", axis=1)
        columns_to_reanme = list(dbh.columns)
        for column in ["date", "province", "region"]:
            columns_to_reanme.remove(column)
        seg_dbh_list[seg_num].rename(
            columns={x: "{}_{}".format(provider, x) for x in columns_to_reanme},
            inplace=True,
        )
        seg_num += 1

    for seg_dbh in seg_dbh_list:
        db = db.merge(seg_dbh, on=["date", "region", "province"])

    loggr.info("Dropping rows with any missing data")
    db.dropna(axis=0, how="any", inplace=True)
    loggr.info("Loading current conditions")
    seg_dbc_list = [0] * 3
    seg_num = 0
    for days_back in range(1, 4):
        seg_dbc_list[seg_num] = dbc.copy()
        columns_to_reanme = list(dbc.columns)
        for column in ["date", "province", "region"]:
            columns_to_reanme.remove(column)
        seg_dbc_list[seg_num].date = (
            seg_dbc_list[seg_num]
            .date.apply(
                lambda x: get_date_object(x) + datetime.timedelta(days=days_back)
            )
            .apply(str)
        )
        seg_dbc_list[seg_num].rename(
            columns={x: "{}_T{}".format(x, days_back) for x in columns_to_reanme},
            inplace=True,
        )
        seg_num += 1
    for seg_dbc in seg_dbc_list:
        db = db.merge(seg_dbc, on=["date", "region", "province"], how="left")

    loggr.info("Computing rolling average of normal highs (per region)")
    db["month"] = db.date.apply(lambda x: get_date_object(x).month)
    db["day_of_month"] = db.date.apply(lambda x: get_date_object(x).day)
    dba["month"] = dba.date.apply(lambda x: get_date_object(x).month)
    dba["day_of_month"] = dba.date.apply(lambda x: get_date_object(x).day)

    seg_dba_list = []
    for region in dba.region.unique():
        seg_dba_list.append(dba[dba.region == region])

    for seg_dba in seg_dba_list:
        seg_dba["rolling_normal_high"] = (
            seg_dba.normal_high.rolling(
                window=rolling_average_window,
                min_periods=rolling_average_min_periods,
                center=True,
            )
            .mean()
            .round(1)
        )

    loggr.info("Merging normal high data into master_db")
    dba_rolled = pd.concat(seg_dba_list, ignore_index=True).drop(
        ["date", "province"], axis=1
    )
    db = db.merge(dba_rolled, on=["month", "day_of_month", "region"], how="left").drop(
        ["month", "day_of_month"], axis=1
    )

    loggr.info("Computing mean columns")
    for reading in ["high", "low"]:
        for T in ["1", "2", "3"]:
            db["mean_{}_T{}".format(reading, T)] = db[
                ["{}_{}_T{}".format(x, reading, T) for x in fc_providers]
            ].apply(np.mean, axis=1)

    loggr.info("Computing average deltas")
    delta_req = [
        "TWN_high",
        "EC_high",
        "TWN_high_T1",
        "EC_high_T1",
        "TWN_high_T2",
        "EC_high_T2",
        "TWN_high_T3",
        "EC_high_T3",
        "mean_high_T1",
        "mean_high_T2",
        "mean_high_T3",
    ]
    for label in delta_req:
        db["{}_delta".format(label)] = db[label] - db.rolling_normal_high

    loggr.info("Merging geocoded data into master_db")
    db = db.merge(dbll, on=["region", "province"], how="left")

    # not sure why but this is necessary for the code not to crash
    db = db.reset_index()

    db = pd.merge(db, pd.get_dummies(db, columns=["province"]))
    for column in [
        "current_wind_direction_T1",
        "current_wind_direction_T2",
        "current_wind_direction_T3",
    ]:
        db = pd.merge(db, pd.get_dummies(db, columns=[column]))
    db.drop(
        [
            "current_wind_direction_T1",
            "current_wind_direction_T2",
            "current_wind_direction_T3",
        ],
        axis=1,
        inplace=True,
    )

    if region_efficient:
        loggr.info("Shrinking data down to South portion of country that is dense in reports")
        db = shrink_regions(db)
        loggr.debug("Number of rows in master_db: {}".format(len(db)))

    loggr.info("Dropping a few columns that will be incompatible with ML training")
    keyword_to_remove = ["current_cond_time", "region_code", "Unnamed", "index"]
    for keyword in keyword_to_remove:
        db = db[db.columns.drop(list(db.filter(regex=keyword)))]
    loggr.debug("Number of columns in master_db: {}".format(len(list(db.columns))))

    if len(np.isnan(db.current_temp_T1)) / len(db) > 0.5:
        loggr.info("Dropping the current conditions columns (feture too young)")
        db = db[db.columns.drop(list(db.filter(regex='current')))]
        loggr.debug("Number of columns in master_db: {}".format(len(list(db.columns))))

    loggr.info("Dropping completely NaN columns")
    all_columns = db.columns
    db.dropna(axis=1, how="all", inplace=True)
    removed_columns = []
    for column in all_columns:
        if column not in db.columns:
            removed_columns.append(column)
    loggr.debug("Dropped: {}".format(removed_columns))
    loggr.debug("Number of columns in master_db: {}".format(len(list(db.columns))))
    
    loggr.info("Dropping rows with any NaN data")
    db.dropna(axis=0, how="any", inplace=True)
    loggr.debug("Number of rows in master_db: {}".format(len(db)))
    
    loggr.info("Dropping any duplicate rows")
    db.drop_duplicates(inplace=True)
    loggr.debug("Number of rows in master_db: {}".format(len(db)))

    loggr.info("Wrangling complete. Saving master_db.csv to disk.")
    db.to_csv("{}/Data/master_db.csv".format(PATH), index=False)
