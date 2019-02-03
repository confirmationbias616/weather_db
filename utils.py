import numpy as np
import pandas as pd
import logging
import datetime
import sys
import os
import argparse
import requests
from bs4 import BeautifulSoup
import time
import random


PATH = os.path.dirname(os.path.abspath(__file__))

loggr = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - line " +
        "%(lineno)d"
    )
)
loggr.addHandler(log_handler)


def str_date_gen(y1=False, y2=False, m1=1, m2=12, d1=1, d2=31):
    """Returns a generator of valid date strings in the format "yyyy-mm-dd".

    Useful when used in conjunction with `.isin()` method to filter Dataframes 
    on the `date` column, which is a string dtype. Returns all valid dates in current 
    year by default.
    \n.
    """

    y1 = datetime.datetime.now().year if y1 is False else y1
    y2 = datetime.datetime.now().year if y2 is False else y2

    def get_date(y, m, d):
        try:
            return str(datetime.date(y, m, d))
        except ValueError:
            return 'skip'
    return (
        get_date(y,m,d) for y in range(
            y1, y2 + 1) for m in range(
            m1, m2 + 1) for d in range(
            d1, d2 + 1) if get_date(y, m, d) != 'skip')

today = str(datetime.datetime.now().date())
yesterday = str(datetime.datetime.now().date() - datetime.timedelta(days=1))
today = "{}-{}-{}".format(today[:4], today[5:7], today[8:10])
yesterday = "{}-{}-{}".format(yesterday[:4], yesterday[5:7], yesterday[8:10])

def erase_today():
    for filename, day in zip(["forecast_db", "history_db"], [today, yesterday]):
        df = pd.read_csv("{}/Data/{}.csv".format(PATH, filename))
        df = df[df["date"] != day]
        df.to_csv("{}/Data/{}.csv".format(PATH, filename), index=False)
    loggr.info("today's data collection has been erased.")


def ETL_regions():
    regions = pd.DataFrame(columns=["province", "region", "EC_region_code"])

    # province_dict has EC province labels as keys and TWN province labels as
    # values
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

    for EC_province in list(province_dict.keys()):
        empty_code = 0
        for EC_region_code in range(1, 500):
            EC_url = "https://weather.gc.ca/city/pages/{}-{}_metric_e.html"
            response = requests.get(EC_url.format(EC_province, EC_region_code))
            if response.status_code == 200:
                html = response.content
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    EC_region_name = soup.find("h1", {"property": "name"}).get_text()[
                        :-5
                    ]
                    regions = regions.append(
                        {
                            "province": province_dict[EC_province],
                            "region": EC_region_name,
                            "EC_region_code": EC_region_code,
                        },
                        ignore_index=True,
                    )
                except AttributeError:
                    pass
                # reset empty-code count
                empty_code = 0
            else:
                empty_code += 1
            if empty_code > 10:
                break
    regions["EC_info"] = regions["province"] + "!" + regions["region"]

    # fucntion that will be used to apply row by row on the EC regions dataframe to fill
    # in TWN codes
    def find_TWN_code(EC_info):

        TWN_prov = EC_info.split("!")[0]
        EC_name = EC_info.split("!")[1]
        if "(" in EC_name:
            return np.nan
        TWN_name = (
            EC_name
        )  # creating TWN copy of region name so we can manipulate to make relevant to
           # TWN nomenclature
        TWN_name = TWN_name.replace(" ", "-")
        TWN_name = TWN_name.replace("'", "")
        TWN_url = "https://www.theweathernetwork.com/ca/weather/{}/{}"
        response = requests.get(TWN_url.format(TWN_prov, TWN_name))
        if response.status_code == 200:
            html = response.content
            soup = BeautifulSoup(html, "html.parser")
            TWN_region_code = (
                soup.find("div", {"id": "prebid", "class": "module"})
                .find_next("script")
                .find_next("script")
                .find_next("script")
                .get_text()
                .split(";")[1][-5:-1]
            )
            print("TWN name: {}".format(TWN_name))
            print("region code: {}".format(TWN_region_code))
            print("\n")
            return TWN_region_code
        time.sleep(random.randint(3, 5))
    regions["TWN_region_code"] = regions["EC_info"].apply(lambda x: find_TWN_code(x))
    regions.drop("EC_info", axis=1, inplace=True)

    def int_or_nan(x):
        if type(x) == str:
            try:
                return int(x)
            except ValueError:
                return np.nan
        else:
            return np.nan

    regions["TWN_region_code"] = regions["TWN_region_code"].apply(
        lambda x: int_or_nan(x)
    )
    regions.dropna(axis=0, inplace=True)
    regions["TWN_region_code"] = (
        regions["TWN_region_code"]
        .apply(lambda x: x[:-2])
        .apply(int)
        .apply(str)
        .apply(lambda x: x.zfill(4))
    )
    regions = regions.reset_index().drop("index", axis=1)
    regions.to_csv("{}/Data/region_codes.csv".format(PATH))


def geocode():
    API_KEY = "GET_YOUR_OWN"
    loggr.info(datetime.datetime.now())
    regions = pd.read_csv("{}/Data/region_codes.csv".format(PATH)).drop(
        "Unnamed: 0", axis=1, errors='ignore'
    )
    regions["prov_region"] = regions["province"].map(str) + "_" + regions["region"]

    def geocodify(prov_region):
        url = "https://maps.googleapis.com/maps/api/geocode/json?address={}+{}+Canada&key={}"
        province, region = prov_region.split("_")
        response = requests.get(url.format(region, province, API_KEY)).json()
        location = response["results"][0]["geometry"]["location"]
        lat = location["lat"]
        lng = location["lng"]
        return str(lat) + "_" + str(lng)

    regions["geocode"] = regions["prov_region"].apply(lambda x: geocodify(x))
    regions["latitude"] = regions["geocode"].apply(lambda x: float(x.split("_")[0]))
    regions["longitude"] = regions["geocode"].apply(lambda x: float(x.split("_")[1]))
    regions.drop(["prov_region", "geocode"], axis=1, inplace=True)
    loggr.info(regions)
    regions.to_csv("{}/Data/region_codes.csv".format(PATH))
    loggr.info(datetime.datetime.now())

def elevate():
    API_KEY = "GET_YOUR_OWN"
    rc = pd.read_csv("{}/Data/region_codes.csv".format(PATH)).drop(
        "Unnamed: 0", axis=1, errors='ignore'
    )
    url = "https://maps.googleapis.com/maps/api/elevation/json?locations={},{}&key={}"
    def get_elevation(latitude, longitude):
        response = requests.get(url.format(latitude, longitude, API_KEY)).json()
        elevation = round(response['results'][0]['elevation'],1)
        print(elevation)
        return elevation
    rc['elevation'] = rc.apply(lambda row: get_elevation(row['latitude'], row['longitude']), axis=1)
    rc.to_csv("{}/Data/region_codes.csv".format(PATH))
    loggr.info(datetime.datetime.now())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tools")
    parser.add_argument(
        "tool",
        nargs="?",
        action="store",
        help="specify which tool you want to use as argument (options: erase_today",
    )
    args = parser.parse_args()
    if args.tool == "erase_today":
        erase_today()
    elif args.tool == "ETL_regions":
        ETL_regions()
    elif args.tool == "geocode":
        geocode()
    elif args.tool == "elevate":
        elevate()
    else:
        loggr.info("No function specified. See help for more instructions.")
