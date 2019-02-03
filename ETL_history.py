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

now = datetime.datetime.now()
region_codes = pd.read_csv("{}/Data/region_codes.csv".format(PATH)).drop(
    "Unnamed: 0", axis=1, errors="ignore"
)
providers = ("TWN", "EC")
readings = {0: "high", 1: "low", 2: "precipitation"}

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
    now = datetime.datetime.now()
    yesterday = now.date() - datetime.timedelta(days=1)
    TWN_region_code = region_codes[
        (region_codes["province"] == prov) & (region_codes["region"] == region)
    ].iloc[0]["TWN_region_code"]
    url = "https://www.theweathernetwork.com/api/historical/ca{}{}/{}/{}/C/metric".format(
        province_dict[prov],
        str(TWN_region_code).zfill(4),
        yesterday.month,
        yesterday.year,
    )
    while True:
        try:
            response = requests.get(url).json()
            break
        except json.decoder.JSONDecodeError:
            loggr.info(
                "For some reason the JSON response was bad. Retrying this code..."
            )
    TWN_data = [None] * len(readings)
    TWN_translation = {0: "temperatureMax", 1: "temperatureMin", 2: "precip"}
    now = datetime.datetime.now()
    try:
        fc_response = response[str(yesterday.day)]
        if fc_response["rdata"]["year"] is "":
            loggr.info(
                "TWN code {} does not have historical data for this"
                "region".format(str(TWN_region_code).zfill(4))
            )
            return [np.nan] * 3
        else:
            for i in range(len(TWN_translation)):
                try:
                    TWN_data[i] = float(fc_response["rdata"][TWN_translation[i]])
                except ValueError:
                    loggr.info(
                        'TWN code {} returned value of "{}" for'
                        'reading "{}"'.format(
                            str(TWN_region_code).zfill(4),
                            TWN_data[i],
                            TWN_translation[i],
                        )
                    )
                    TWN_data[i] = np.nan
    except (KeyError, TypeError):
        loggr.critical(
            "TWN code {} is a bad region code? JSON response"
            'received from the region code: "{}"'.format(
                str(TWN_region_code).zfill(4), response
            )
        )
        pass
    return TWN_data


def get_EC(prov, region, readings):
    now = datetime.datetime.now()
    yesterday = now.date() - datetime.timedelta(days=1)
    url = "https://weather.gc.ca/city/pages/{}-{}_metric_e.html"
    EC_region_code = region_codes[
        (region_codes["province"] == prov) & (region_codes["region"] == region)
    ].iloc[0]["EC_region_code"]
    response = requests.get(url.format(province_dict[prov], EC_region_code))
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    hist_link = (
        soup.find("summary", {"id": "yesterday"})
        .find_next_sibling()
        .find("li")
        .find("a")
        .attrs["href"]
    )
    response = requests.get(hist_link)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    hist_year = int(
        soup.find("div", {"id": "dynamicDataTable"}).find("caption").get_text()[-4:]
    )
    EC_data = [0 for _ in range(len(readings))]
    if hist_year == yesterday.year:
        try:
            table_row = soup.find(
                "abbr", text=str("%02d" % (yesterday.day,))
            ).find_all_next("td")
        except AttributeError:
            loggr.info(
                "EC code {} does not currently have data for "
                "yesterday".format(EC_region_code)
            )
            EC_data = [np.nan] * 3
            return [np.nan] * 3
        for i, j in zip(range(3), [0, 1, 7]):
            try:
                EC_data[i] = float(table_row[j].get_text())
            except ValueError:
                if table_row[j].get_text() == "LegendMM":
                    loggr.info(
                        "EC code {} reports a missing value for "
                        'reading "{}"'.format(EC_region_code, readings[i])
                    )
                    pass
                else:
                    loggr.warning(
                        'EC code {} returned value of "{}" for '
                        'reading "{}"'.format(
                            EC_region_code, table_row[j].get_text(), readings[i]
                        )
                    )
                    pass
                EC_data[i] = np.nan
        return EC_data
    else:
        loggr.info(
            "EC code {} only has historical data for "
            "year {}".format(EC_region_code, hist_year)
        )
        return [np.nan] * 3


history_db = pd.read_csv("{}/Data/history_db.csv".format(PATH))
if history_db["date"].iloc[-1] == (now.date() - datetime.timedelta(days=1)).strftime(
    "%Y-%m-%d"
):
    loggr.warning("Data already collected for today. Process terminated.")
    pass
else:
    no_of_regions = len(region_codes["TWN_region_code"])
    loggr.info(
        "starting to extract history for 1st of " "{} regions...".format(no_of_regions)
    )
    for j in range(no_of_regions):
        provider_history = [[] for _ in range(len(providers))]
        i = 0
        for op in (get_TWN, get_EC):
            data = op(region_codes.province[j], region_codes.region[j], readings)
            provider_history[i] = pd.DataFrame(
                {
                    "date": datetime.datetime.now().date() - datetime.timedelta(days=1),
                    "time": datetime.datetime.now().time(),
                    "provider": providers[i],
                    "province": region_codes.iloc[j]["province"],
                    "region": region_codes.iloc[j]["region"],
                    "high": pd.Series(data[0]),
                    "low": pd.Series(data[1]),
                    "precipitation": pd.Series(data[2]),
                }
            )
            i += 1
        history_yesterday = pd.concat(provider_history, sort=True)
        j += 1
        loggr.info("extracted history for region #{}".format(j))
        history_db = history_db.append(history_yesterday, ignore_index=True)
    history_db.to_csv("{}/Data/history_db.csv".format(PATH), index=False)
