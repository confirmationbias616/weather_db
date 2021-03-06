import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
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
    "Unnamed: 0", axis=1, errors="ignore"
)
providers = ("TWN", "EC")
fc_days = 5
readings = {
    0: "high_forecast",
    1: "low_forecast",
    2: "day_pop_forecast",
    3: "night_pop_forecast",
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
    "newfoundland-and-labrador": "nl",
    "yukon": "yt",
    "nunavut": "nu",
    "northwest-territories": "nt",
}


def get_TWN(prov, region, readings, fc_days):
    url = "https://www.theweathernetwork.com/api/data/ca{}{}"
    TWN_region_code = region_codes[
        (region_codes["province"] == prov) & (region_codes["region"] == region)
    ].iloc[0]["TWN_region_code"]
    max_retries = 20
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(
                url.format(province_dict[prov], str(TWN_region_code).zfill(4))
            ).json()
            if len(response) < fc_days - 1:
                loggr.warning("TWN returned bad data for this region. Retrying...")
                retries += 1
                continue
        except json.decoder.JSONDecodeError:
            loggr.warning(
                "For some reason the JSON response was bad. Retrying this code..."
            )
            retries += 1
            continue
        TWN_data = [None] * len(readings)
        TWN_translation = {0: "tmac", 1: "tmic", 2: "pdp", 3: "pnp"}
        for i in range(len(TWN_translation)):
            fc_response = response["fourteendays"]["periods"]
            TWN_data[i] = [
                int(fc_response[day][TWN_translation[i]]) for day in range(0, fc_days)
            ]
        retries += 1
        return TWN_data
    loggr.warning(
        "Exceeded retry limit. logging all NaNs and moving on to the next code."
    )
    return [[np.nan] * fc_days] * len(readings)


def get_EC(prov, region, readings, fc_days):
    url = "https://weather.gc.ca/city/pages/{}-{}_metric_e.html"
    EC_region_code = region_codes[
        (region_codes["province"] == prov) & (region_codes["region"] == region)
    ].iloc[0]["EC_region_code"]
    response = requests.get(url.format(province_dict[prov], EC_region_code))
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    EC_data = [[] for _ in range(len(readings))]
    forecast_table = soup.find("div", {"class": "div-table"})
    try:
        forecast_days = forecast_table.find_all("div", {"class": "div-column"})[
            1 : (fc_days + 1)
        ]
    except AttributeError:
        return [[np.nan]*5]*4
    for col_content in forecast_days:
        try:
            EC_data[0].append(
                int(
                    col_content.find(
                        "span", {"class": "high wxo-metric-hide"}
                    ).get_text()[:-2]
                )
            )
        except ValueError:
            EC_data[0].append(0)
        try:
            EC_data[1].append(
                int(
                    col_content.find(
                        "span", {"class": "low wxo-metric-hide"}
                    ).get_text()[:-2]
                )
            )
        except ValueError:
            EC_data[1].append(0)
        for reading, itr in zip([EC_data[2], EC_data[3]], (0, 1)):
            try:
                result_str = col_content.find_all(
                    "p", {"class": "mrgn-bttm-0 pop text-center"}
                )[itr]
                reading.append(int(result_str.get_text()[:-1]))
            except ValueError:
                reading.append(0)
    return EC_data


forecast_db = pd.read_csv("{}/Data/forecast_db.csv".format(PATH))
if forecast_db["date"].iloc[-1] == datetime.datetime.now().date().strftime("%Y-%m-%d"):
    loggr.warning("Data already collected for today. Process terminated.")
    pass
else:
    no_of_regions = len(region_codes["TWN_region_code"])
    loggr.info(
        "starting to extract forecast for 1st of " "{} regions...".format(no_of_regions)
    )
    for j in range(no_of_regions):
        provider_forecasts = [[] for _ in range(len(providers))]
        i = 0
        for op in (get_TWN, get_EC):
            data = op(
                region_codes.province[j], region_codes.region[j], readings, fc_days
            )
            provider_forecasts[i] = pd.DataFrame(
                {
                    "date": datetime.datetime.now().date(),
                    "time": datetime.datetime.now().time(),
                    "provider": providers[i],
                    "province": region_codes.iloc[j]["province"],
                    "region": region_codes.iloc[j]["region"],
                    "day": list(range(1, fc_days + 1)),
                    "high": pd.Series(data[0]),
                    "low": pd.Series(data[1]),
                    "day_pop": pd.Series(data[2]),
                    "night_pop": pd.Series(data[3]),
                }
            )
            i += 1
        forecast_today = pd.concat(provider_forecasts)
        j += 1
        loggr.info("extracted forecast for region #{}".format(j))
        forecast_db = forecast_db.append(forecast_today, ignore_index=True)
    forecast_db.to_csv("{}/Data/forecast_db.csv".format(PATH), index=False)
