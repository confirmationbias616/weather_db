import datetime
import requests
import pandas as pd
import sys
import os
import logging
import json


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

region_codes = pd.read_csv("{}/Data/region_codes.csv".format(PATH)).drop(
    "Unnamed: 0", axis=1
)
providers = "TWN"
readings = {
    0: "current_temp",
    1: "current_temp_feels",
    2: "current_pressure",
    3: "current_wind_speed",
    4: "current_wind_direction",
}

# province_dict has province labels from region_codes.csv as keys and TWN
# province labels as values
province_dict = {
    "nova-scotia": "ns",
    "prince-edward-island": "pe",
    "new-brunswick": "nb",
    "quebec": "qc",
    "ontario": "on",
    "manitoba": "mb",
    "saskatchewan": "sk",
    "alberta": "ab",
    "british-columbia": "bc",
}


def get_TWN(prov, region, readings):
    url = "https://www.theweathernetwork.com/api/data/ca{}{}"
    TWN_region_code = region_codes[
        (region_codes["province"] == prov) & (region_codes["region"] == region)
    ].iloc[0]["TWN_region_code"]
    while True:
        try:    
            response = requests.get(
                url.format(province_dict[prov], str(TWN_region_code).zfill(4))
            ).json()
            break
        except json.decoder.JSONDecodeError:
            loggr.info("For some reason the JSON response was bad. Retrying this code...")
    TWN_data = [None] * len(readings)
    TWN_translation = {0: "t", 1: "f", 2: "p", 3: "w", 4: "wd"}
    try:
        for i in range(len(TWN_translation)):
            fc_response = response["obs"]
            try:
                TWN_data[i] = int(fc_response[TWN_translation[i]])
            except ValueError:
                try:
                    TWN_data[i] = float(fc_response[TWN_translation[i]])
                except ValueError:
                    TWN_data[i] = str(fc_response[TWN_translation[i]])
        return TWN_data
    except KeyError:
        loggr.warning("bad region code?")
        loggr.warning(
            "JSON response we got from the " "region code: \n{}".format(response)
        )           


current_db = pd.read_csv("{}/Data/current_db.csv".format(PATH))
if history_db["date"].iloc[-1] == (now.date() - datetime.timedelta(days=1)).strftime(
    "%Y-%m-%d"
):
    loggr.warning("Data already collected for today. Process terminated.")
    pass
else:
    no_of_regions = len(region_codes["TWN_region_code"])
    loggr.info(
        "starting to extract current conditions for 1st of "
        "{} regions...".format(no_of_regions))
    for j in range(no_of_regions):
        data = get_TWN(region_codes.province[j], region_codes.region[j], readings)
        loggr.info(data)
        current_db = current_db.append(
            {
                "date": datetime.datetime.now().date(),
                "province": region_codes.iloc[j]["province"],
                "region": region_codes.iloc[j]["region"],
                "current_cond_time": datetime.datetime.now().time(),
                "current_temp": data[0],
                "current_temp_feels": data[1],
                "current_pressure": data[2],
                "current_wind_speed": data[3],
                "current_wind_direction": data[4],
            }, ignore_index=True
        )
        loggr.info("extracted current conditions for region #{}".format(j+1))
    current_db.to_csv("{}/Data/current_db.csv".format(PATH), index=False)
