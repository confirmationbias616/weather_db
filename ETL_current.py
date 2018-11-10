import datetime
import requests
import pandas as pd
import sys
import os
import logging


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
    0: "temp",
    1: "temp_feels",
    2: "pressure",
    3: "wind_speed",
    4: "wind_direction",
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
    response = requests.get(
        url.format(province_dict[prov], str(TWN_region_code).zfill(4))
    ).json()
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
    except KeyError:
        loggr.warning("bad region code?")
        loggr.warning(
            "JSON response we got from the " "region code: \n{}".format(response)
        )
    return TWN_data


loaded_current_db = pd.read_csv("{}/Data/current_db.csv".format(PATH))
current_db = loaded_current_db.drop(loaded_current_db.index)
no_of_regions = len(region_codes["TWN_region_code"])
loggr.info(
    "starting to extract current conditions for 1st of "
    "{} regions...".format(no_of_regions)
)
for j in range(no_of_regions):
    data = get_TWN(region_codes.province[j], region_codes.region[j], readings)
    loggr.info(data)
    current_db = current_db.append(
        {
            "date": datetime.datetime.now().date(),
            "current_cond_time": datetime.datetime.now().time(),
            "province": region_codes.iloc[j]["province"],
            "region": region_codes.iloc[j]["region"],
            "temp": data[0],
            "temp_feels": data[1],
            "pressure": data[2],
            "wind_speed": data[3],
            "wind_direction": data[4],
        }, ignore_index=True
    )
    loggr.info("extracted current conditions for region #{}".format(j))
current_db = loaded_current_db.append(current_db, ignore_index=True)
current_db.to_csv("{}/Data/current_db.csv".format(PATH), index=False)
