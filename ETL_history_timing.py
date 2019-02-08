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
yesterday = now.date() - datetime.timedelta(days=1)
region_codes = pd.read_csv("{}/Data/region_codes.csv".format(PATH)).drop(
    "Unnamed: 0", axis=1, errors="ignore"
)

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

def get_EC_high_data(prov, region):
    url = "https://weather.gc.ca/city/pages/{}-{}_metric_e.html"
    EC_region_code = region_codes[
        (region_codes["province"] == prov) & (region_codes["region"] == region)
    ].iloc[0]["EC_region_code"]
    try:
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
        hist_day_link = (
            soup.find("table", {"class": "table table-striped table-hover data-table"})
            .find("tbody")
            .find("abbr", {"title": yesterday.strftime("%B %d, %Y").replace(" 0", " ")})
            .find("a")
            .attrs["href"]
        )
        response = requests.get("http://climate.weather.gc.ca"+hist_day_link)
        html = response.content
        soup = BeautifulSoup(html, "html.parser")
        time_table = soup.find("tbody")
        hourly_log = time_table.find("tr").find_next_siblings()
    except AttributeError:
        return pd.Series(
            [prov, region] + [np.nan]*4, 
            index=["province", "region", "high_24hr_time", "high_24hr_temp", "high_12hr_time", "high_12hr_temp"]
        )
    log_tuples_24hr = []
    for log_entry in hourly_log:
        time = int(log_entry.find("td").get_text()[:2])
        temp = log_entry.find("td").find_next_sibling().get_text()
        try:
            temp = float(temp)
        except ValueError:
            temp = np.nan
        log_tuples_24hr.append((time,temp))
    log_tuples_12hr = log_tuples_24hr[6:19]
    high_24hr_time, high_24hr_temp = sorted(log_tuples_24hr, key=lambda tup: tup[1], reverse=True)[0]
    high_12hr_time, high_12hr_temp = sorted(log_tuples_12hr, key=lambda tup: tup[1], reverse=True)[0]
    loggr.info(
        "high_24hr_time:{} ".format(high_24hr_time)+
        "high_24hr_temp:{} ".format(high_24hr_temp)+
        "high_12hr_time:{} ".format(high_12hr_time)+
        "high_12hr_temp:{} ".format(high_12hr_temp)
    )
    loggr.info("extracted current conditions for {}, {}".format(region, prov))
    return pd.Series(
        [prov, region, high_24hr_time, high_24hr_temp, high_12hr_time, high_12hr_temp], 
        index=["province", "region", "high_24hr_time", "high_24hr_temp", "high_12hr_time", "high_12hr_temp"]
    )

htdb = pd.read_csv("{}/Data/history_timing_db.csv".format(PATH))
if htdb["date"].iloc[-1] == (now.date() - datetime.timedelta(days=1)).strftime(
    "%Y-%m-%d"
):
    loggr.warning("Data already collected for today. Process terminated.")
    pass
else:
    loggr.info(
        "starting to extract current conditions for 1st of "
        "{} regions...".format(len(region_codes))
    )
    df = region_codes.apply(lambda row: get_EC_high_data(row['province'], row['region']), axis=1)
    df["high_24hr_time"] = pd.to_numeric(df["high_24hr_time"], downcast='integer', errors='coerce')
    df["high_12hr_time"] = pd.to_numeric(df["high_12hr_time"], downcast='integer', errors='coerce')
    df['date'] = yesterday
    df['provider'] = "EC"
    htdb = htdb.append(df, ignore_index=True)
    htdb.to_csv("{}/Data/history_timing_db.csv".format(PATH), index=False)
