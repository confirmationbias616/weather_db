import logging
import sys
import pandas as pd
import datetime
import os


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line %(lineno)d"
    )
)
loggr.addHandler(log_handler)
loggr.setLevel(logging.INFO)

province_dict = {
    "ns": "nova-scotia",
    "pe": "prince-edward-island",
    "nb": "new-brunswick",
    "qc": "quebec",
    "on": "ontario",
    "mb": "manitoba",
    "sk": "saskatchewan",
    "ab": "alberta",
    "bc": "british-columbia",
}


def wrangle(
    time_span=10,
    rolling_average_window=30,
    rolling_average_min_periods=1,
    date_efficient=True,
    region_efficient=False,
    **kwargs
):
    def shrink_dates(df):
        pillow = rolling_average_window + time_span + 10
        today_date = datetime.date(int(today[:4]), int(today[5:7]), int(today[8:]))
        return df[
            (df["date"] < str(today_date + datetime.timedelta(days=pillow)))
            & (df["date"] > str(today_date - datetime.timedelta(days=pillow)))
        ]

    def shrink_regions(df):
        return df[
            ((df["longitude"] * -0.09) + 43 > df["latitude"]) & (df["latitude"] < 53)
        ]

    loggr.info("Starting to Wrangle data into fresh version of master_db.csv...")
    try:
        today = kwargs["target_date"]
    except KeyError:
        today = str(datetime.datetime.now().date())
    loggr.info("Loading history data")
    dbh = pd.read_csv("{}/Data/history_db.csv".format(PATH))
    if date_efficient:
        dbh = shrink_dates(dbh)
    dbh.drop("time", axis=1, inplace=True)
    dbh["date"] = dbh["date"].apply(
        lambda x: datetime.date(int(x[:4]), int(x[5:7]), int(x[8:]))
    )
    dbh = dbh.set_index("provider")

    loggr.info("Wrangling history data")
    dbh_per_provider = []
    for provider in ["TWN", "EC"]:
        dbh_P = dbh.xs(provider).reset_index()
        dbh_P_c = dbh_P[["date", "high", "region", "province"]].copy()
        dbh_P_c["date_tomorrow"] = dbh_P_c["date"].apply(
            lambda x: x + datetime.timedelta(days=2)
        )
        dbh_P_c["high_2ago"] = dbh_P_c["high"]
        dbh_P_c.drop(["date", "high"], axis=1, inplace=True)
        dbh_P = dbh_P.merge(
            dbh_P_c,
            left_on=["date", "region", "province"],
            right_on=["date_tomorrow", "region", "province"],
            how="outer",
        )
        dbh_P["date"] = dbh_P.apply(
            lambda row: row["date_tomorrow"]
            if type(row["date"]) == float
            else row["date"],
            axis=1,
        )
        dbh_P["provider"] = provider  # need to backfill provider for new rows
        dbh_P.drop("date_tomorrow", axis=1, inplace=True)
        dbh_P.rename(
            index=str,
            columns={
                "high": "{}_high".format(provider),
                "low": "{}_low".format(provider),
                "precipitation": "{}_precipitation".format(provider),
                "high_2ago": "{}_high_2ago".format(provider),
            },
            inplace=True,
        )
        dbh_per_provider.append(dbh_P.drop("provider", axis=1))
    dbh_flat = dbh_per_provider[0]
    for dbh_P in dbh_per_provider[1:]:
        dbh_flat = dbh_flat.merge(dbh_P, on=["date", "region", "province"], how="left")
    dbh_flat["year"] = dbh_flat["date"].apply(lambda x: x.year).apply(str)
    dbh_flat["month"] = dbh_flat["date"].apply(lambda x: x.month).apply(str)
    dbh_flat["day"] = dbh_flat["date"].apply(lambda x: x.day).apply(str)
    dbh_flat.drop("date", axis=1, inplace=True)

    loggr.info("Loading forecast data")
    dbf = pd.read_csv("{}/Data/forecast_db.csv".format(PATH))
    if date_efficient:
        dbf = shrink_dates(dbf)
    dbf = dbf.set_index(["date", "provider", "day", "region", "province"])
    dbf_TWN = dbf.xs("TWN", level="provider")
    dbf_EC = dbf.xs("EC", level="provider")

    loggr.info("Wrangling forecast data")
    dbf_date_shift = [0] * 10
    j = 0
    for provider, data in zip(["TWN", "EC"], [dbf_TWN, dbf_EC]):
        for i in range(5):
            dbf_date_shift[i + j] = data.xs(i + 1, level="day")
            dbf_date_shift[i + j] = (
                dbf_date_shift[i + j].reset_index().drop("time", axis=1)
            )
            dbf_date_shift[i + j]["date"] = dbf_date_shift[i + j]["date"].apply(
                lambda date: datetime.datetime.strptime(date, "%Y-%m-%d")
                + datetime.timedelta(days=i + 1)
            )
            dbf_date_shift[i + j].rename(
                index=str,
                columns={
                    "high": "{}_high_T{}".format(provider, i + 1),
                    "low": "{}_low_T{}".format(provider, i + 1),
                    "day_pop": "{}_day_pop_T{}".format(provider, i + 1),
                    "night_pop": "{}_night_pop_T{}".format(provider, i + 1),
                },
                inplace=True,
            )
            dbf_date_shift[i + j]["year"] = (
                dbf_date_shift[i + j]["date"].apply(lambda x: x.year).apply(str)
            )
            dbf_date_shift[i + j]["month"] = (
                dbf_date_shift[i + j]["date"].apply(lambda x: x.month).apply(str)
            )
            dbf_date_shift[i + j]["day"] = (
                dbf_date_shift[i + j]["date"].apply(lambda x: x.day).apply(str)
            )
            dbf_date_shift[i + j].reset_index(inplace=True)
            dbf_date_shift[i + j].set_index(
                ["year", "month", "day", "region", "province"], inplace=True
            )
        j += 5
    dbf_flat = dbf_date_shift[0]
    for item in dbf_date_shift[1:]:
        dbf_flat = dbf_flat.merge(
            item.drop(["date"], axis=1),
            on=["year", "month", "day", "region", "province"],
            how="outer",
        )
    dbf_flat.drop(["index_x", "index_y"], axis=1, inplace=True)
    dbf_flat.reset_index(inplace=True)
    dbh_flat.region = dbh_flat.region.apply(str)
    dbh_flat.province = dbh_flat.province.apply(str)
    dbh_flat.year = dbh_flat.year.apply(int)
    dbh_flat.month = dbh_flat.month.apply(int)
    dbh_flat.day = dbh_flat.day.apply(int)
    dbf_flat.region = dbf_flat.region.apply(str)
    dbf_flat.province = dbf_flat.province.apply(str)
    dbf_flat.year = dbf_flat.year.apply(int)
    dbf_flat.month = dbf_flat.month.apply(int)
    dbf_flat.day = dbf_flat.day.apply(int)
    dbf_flat.reset_index(inplace=True)
    dbh_flat.reset_index(inplace=True)
    loggr.info("Merging forecast and history data into master_db")
    db = dbf_flat.merge(
        dbh_flat, on=["day", "month", "year", "region", "province"], how="left"
    )
    db.reset_index().drop(["index"], axis=1, inplace=True)

    loggr.info("Loading current conditions")
    
    for days_back in range(1,4):
        dbc = pd.read_csv("{}/Data/current_db.csv".format(PATH))
        variable_dbc_columns = list(dbc.columns)
        for fixed_dbc_variable in ['date', 'province', 'region']:
            variable_dbc_columns.remove(fixed_dbc_variable)
        dbc.date = dbc.date.apply(lambda x: datetime.date(int(x[:4]), int(x[5:7]), int(x[8:])) + datetime.timedelta(days_back)).apply(str)
        dbc.rename(columns={x:x+'_T{}'.format(days_back) for x in list(variable_dbc_columns)})
        dbc["year"] = dbc["date"].apply(lambda x: x[:4]).apply(int)
        dbc["month"] = dbc["date"].apply(lambda x: x[5:7]).apply(int)
        dbc["day"] = dbc["date"].apply(lambda x: x[8:]).apply(int)
        dbc.drop("date", axis=1, inplace=True)
        loggr.info("Merging current conditions into master_db")
        db = db.merge(dbc, on=["year", "month", "day", "region", "province"], how="left")

    loggr.info("Loading normal high data")
    dba = pd.read_csv(
        "{}/Data/dba.csv".format(PATH),
        dtype={
            "year": "int",
            "month": "int",
            "day": "int",
            "region": "str",
            "province": "str",
            "latitude": "float",
            "longitude": "float",
        },
    )
    if date_efficient:
        dba = shrink_dates(dba)
    dba.drop(["Unnamed: 0"], axis=1, inplace=True)
    dba.dropna(inplace=True)
    dba = dba.set_index(["region", "date"])
    dba_roll = dba.drop(dba.index)
    loggr.info("Computiong rolling average of normal highs (per region)")
    for region, dba in dba.groupby(level=0):
        dba["rolling normal high"] = (
            dba["normal high"]
            .rolling(
                window=rolling_average_window,
                min_periods=rolling_average_min_periods,
                center=True,
            )
            .mean()
            .round(1)
        )
        dba_roll = dba_roll.append(dba)
    dba_roll.reset_index(inplace=True)
    dba_roll["year"] = dba_roll["date"].apply(lambda x: x[:4]).apply(int)
    dba_roll["month"] = dba_roll["date"].apply(lambda x: x[5:7]).apply(int)
    dba_roll["day"] = dba_roll["date"].apply(lambda x: x[8:]).apply(int)
    dba_roll["province"] = dba_roll["province"].apply(lambda x: province_dict[x])
    dba_roll.drop(["date", "year"], axis=1, inplace=True)
    loggr.info("Merging normal high data into master_db")
    db = db.merge(dba_roll, on=["month", "day", "region", "province"], how="left")

    loggr.info("Computing average deltas")
    delta_req = [
        "TWN_high_2ago",
        "EC_high_2ago",
        "TWN_high",
        "EC_high",
        "TWN_high_T1",
        "EC_high_T1",
        "TWN_high_T2",
        "EC_high_T2",
        "TWN_high_T3",
        "EC_high_T3",
        "TWN_low_T1",
        "EC_low_T1",
        "TWN_low_T2",
        "EC_low_T2",
        "TWN_low_T3",
        "EC_low_T3",
    ]
    for label in delta_req:
        db["{}_delta".format(label)] = db[label] - db["rolling normal high"]
    loggr.info("Loading geocoded info")
    dbll = pd.read_csv(
        "{}/Data/region_codes.csv".format(PATH),
        dtype={
            "region": "str",
            "province": "str",
            "latitude": "float",
            "longitude": "float",
        },
    ).drop("Unnamed: 0", axis=1)
    loggr.info("Merging geocoded data into master_db")
    db = db.merge(dbll, on=["region", "province"], how="left")

    if region_efficient:
        db = shrink_regions(db)

    db['days_sequence'] = db.reset_index()['index']

    db.to_csv("{}/Data/master_db.csv".format(PATH))
