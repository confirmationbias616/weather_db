# weather_db :cloud:

[![CircleCI](https://circleci.com/gh/confirmationbias616/weather_db.svg?style=svg)](https://circleci.com/gh/confirmationbias616/weather_db)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Forecasting Canada's weather by analyzing top forecasters's data and improving upon their prediction errors. What sets this weather forecasting engine apart from its competitors is that doesn't use meteorology at all — no fancy weather models, no radar projections. Pure data is its only knowledge and ML is the secret sauce. 


###### Primary Goal:
* Predict tomorrow's high with as much accuracy as possible. [Check out my progress on Klipfolio!](https://app.klipfolio.com/published/dcdee1e03d96198ac7b9d659ae29357a/accuracy-improvements)
 * A real success would be to have the lowest prediction errors (MAE) between forecast and recorded highs out of all tracked forecasters.

 
###### Secondary Goals:
* Find out which weather forecaster has the best predictions and under which circumstances.
* Find out which features are the most important to predict the weather without meteorology.
* Find out if rumours of forecaster [wet bias](<https://en.wikipedia.org/wiki/Wet_bias>) are valid.
* Collect a dataset of historical forecasts across the country for whatever new purpose the open source community might find

---

Please understand this project is still work in progress and is very much **incomplete**.  
See [Installation](#Installation) instructions to check out the source code on your machine and maybe even help me out.

Also, see [Progress Log](#ProgressLog) for latest updates on development.

---

## Challenges

### <a name="Drift"></a>Drift 
Given that Canada is a country with 4 seasons, historical daily averages are constantly on the move from one day/week/month to the next. It's unreasonable to expect a machine learning model trained on January data to perform just as well in July. This boils down to the well-known issue of data drift, where the continuous mutation of feature characteristics results in unpredictability of the target variable. In this application, we are always trying to predict tomorrow's wether, which means we are trying to predict an edge case for which the model has been poorly trained on. One way to solve this would be to train the model on a period of time which equally straddles the forecast date. As we've been collecting data for less than 1 year however, this is solution is not yet possible. The only feasible solution for now is to keep the training sets short every new date's prediction model. So far, it seems like a 10-day trailing training period is optimal.

### Micro-Climates
On any given day, the weather across Canada's vast territory tends to vary immensely depending on geographic location. If this is not controlled for, the ML model will be unsuccessful. We solve this issue by pulling in each location's latitude, longitude, and elevation from Google's Geocoding API. This means the ML model can cater its predictions based on geographical information.

### Geographic Sparsity
Given that Canada is sparsely populated, regions included in weather forecasts tend to be clustered and not very well distributed. This challenge is mostly resolved by using the latitude and longitude features, as described above, but there is still an issue with the fact that we have very little data originating from the Northern regions of the country. It might be worthwhile to exclude every datapoint above a certain latitude threshold — even if it means losing the functionality to predict in those regions. For now, Yukon, Northwest Territories, Nunavut, and Newfoundland and Labrador are excluded. Later, I plan on trying out over-sampling of the sparse regions or under-sampling the dense regions to allow these Northern back in the algorithm.

### Timing of Data Availability
Since we are trying to predict next-day weather with the data we have during the current day, it behooves us to receive data in a timely manner. A major component of being able to forecast the weather lies in the concept of [weather persistence](<https://en.wikipedia.org/wiki/Weather_forecasting#Persistence>). Basically this means that the weather tends to maintain itself over short periods of time. The short-term past is a good predictor of the near future. Therefore, gathering current day conditions as a feature set would surely be of massive benefit for the ML model's performance.  **-- Good news, this is now part of the model!**

### Poor Historical Data
The quality of historical data posted by the weather forecasters isn't great. There's a ton of inconsistencies and missing data. This really hurts model performance because the historical data scraped from the forecasting websites is what forms our ground truth.

### Inconsitent Occurence of Daily High
The daily high usually occurs near mid-day. Under special circumstances however, the daily high can occur at any other time. These off-beat daily highs really throw off the model. It might be worth trying out a multitask learning (MLT) neural net to predict both timing and actual value of the daily high. This would add robustness to the model. However, neural nets are very data-hungry and our model can only train on short spans of recent data, [as explained above.](#Drift) Until a whole year's data has been collected, this is probably something that just has to be tolerated.

## End-to-End Machine Learning
Here's a brief description of all the moving parts in this project.

### ETL (Extract-Transform-Load)

#### Extract
Use Requests to scrape websites of tracked forecasters for all available locations in Canada. Current list of tracked forecasters:

* The Weather Network
* Environment Canada

Favour API's where available. If there's no API, default to using BeautifulSoup to parse through relevant info.

Data to retrieve:
* (1,2,3,4,5)-day forecast for the following measurements:
    * High
    * Low
    * Day pop
    * Night pop
    * Total precipitation
* Yesterday’s data:
    * High
    * Low
    * Total precipitation

#### Transform & Load
Use Pandas to store all data into `.csv` files and load back into memory when necessary. Also use Pandas to wrangle the data into suitable shapes for downstream processes.

#### Validate
Every time new data is collected, run a few tests to make sure data collection ran as expected and data is acceptable. Log any exceptions.

#### Train
Use Scikit-Learn to train a random forest model on historical data, using the reported daily high by The Weather Network as a the target variable. Pickle the model for future use in predicting daily highs.

#### Predict
Unpickle appropriate model generated in previous step and use it to predict.

#### Report
Feed prediction data into [Klipfolio](https://app.klipfolio.com/published/dcdee1e03d96198ac7b9d659ae29357a/accuracy-improvements) to track performance and provide a way to explore model shortcomings to alow better planning for future improvements.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Make sure you are running Python3 and have installed all required packages using `pip install -r requirements.txt`

### <a name="Installation"></a>Installation
* Clone this repo to your computer.
* Get into the folder using ```cd weather_db```.
* Run ```mkdir Data Gym```.
* Download contents from each of the following dropbox links and copy them to your local respective newly created directories
    * [Data](https://www.dropbox.com/sh/wh9feldwhj3wagu/AADdnMciGo5Jk8WcIk3m5uTNa?dl=0)
    * [Gym](https://www.dropbox.com/sh/cai718eeplvb8fi/AAAF1Nnz5w098HDgIH0WoWg3a?dl=0)

### Schedule
Currently, I run the following script on a daily schedule, using launchd through a great little app called [LaunchControl](<http://www.soma-zone.com/LaunchControl/>):
* `Daily.py` at 5:00 PM

If you want the same results as me, you should do the same.

## <a name="ProgressLog"></a>Progress Log

---

#### Feb 2, 2019
Since the productionisation of weather_db in early December, I've been casually checking in on my Klipfolio dashboard to monitor the model's predictions across the country. I've been trying to get a feel for quality and usefulness of the generaed output to see if any tweaks are required.

The KPI of beating Environment Canada and The Weather Network on average by 21% and 11% respectively seems rather encouraging at first glance but what happens when we zoom in on specific areas and dates? I started noticing a reccuring pattern of weather_db far outperforming other forecasters when and where the daily high occurs in the first hours of a new day (12AM - 3AM). When the day's high occurs closer to mid-day however, the competition seems to perform much better. 

**So what's going on here?!** Do these forecasters have flawed models and algorithms that just refuse to consider the possibility of a day's high occuring in the wee hours of the night? Are they just off on their timing of temperature drops in the evening caused by the sun's retreat? As it turns out, none of these explanations are right. 

As explained by [Environment Canada's website](https://www.canada.ca/en/environment-climate-change/services/types-weather-forecasts-use/public/guide/bulletins.html#c1), the "Tomorrow Forecast" being issued in the evening only considers the tomorrow's range of 6AM - 6PM when it predicts a daily high. For The Weather Network, I can't find anything on their page explaining forecast timing but I did e-mail their support team to get more info. I suspect they must be using the same structure because their median prediction error is -1.1 degrees, which is similar to Environment Canada (-1.2). Reflecting back on the preliminary EDA phase, looking at summary statistics from the initial web scraping round of data should have sounded alarm bells. I do remember finding these biases strange but chalked it up to probably being a temperature version of the [wet bias](<https://en.wikipedia.org/wiki/Wet_bias>). See [this notebook](https://github.com/confirmationbias616/weather_db/blob/master/Notebooks/Prediction_Error_Bias_Analysis.ipynb) for a short analyis on the last few lines. For the record, weather_db's median is virtually neutral (-0.05), which is as it should be.

Unlike the time period for the forecasters' predicted daily highs, they report *historical* daily highs over the day's full 24-hr period - not just the mid-day 6AM - 6PM period. Since the model is trained on this reported *historical* data as ground truth, it therefore tries to predict the daily high for the full 24-hr period. This results in a mismatch between the model's daily high predictions and its competitors' daily high predictions.

 This recent discovery generates 2 serious issues:
* Comparing perfomance of weather_db to that of its competitors is not fair because they technically aren't predicting the same thing. In fact, comparing Enivronment Canada's predictions to its own historical records doesn't even make sense, for this exact same reason. This leads me to wonder: are they even tracking their own performance?
* The model could easily become biased towards predicting too high in weather patterns where nightly highs are a common occurence, which would then backfire when daily highs start occuring at mid-day again.

This leaves the project at a fork in the road, stuck between 2 possible paths it could take to realign the data, predictions, and KPI's in a more meaningful way:
1. Keep everyhting generally the same but stop claiming a certain outperformance percentage over competition. The forecasters' limited-range predictions are still great features to be used by the ML model - they just aren't exactly comparable to the model's target variable and cannot be compared as such. Sticking with this path, we could still add 2 features to our trainging data: time of daily high (continuous numerical feature) and whether it falls before mid-day range, during, or after (categorical feature). 
2. Tweak the ETL & data wrangling process so that the recorded historical daily high is actually just taken from the 6AM-6PM time period. This hourly breakdown of the actual temperature is available through Environment Canada's website. Modifying the target variable of the model's training dataset in this way would in effect cause it to mimic its competitor's behaviour (but hopefully with better performance!). We would finally be able to tell if weather_db can outperform its competition!

This fork in the road calls for 2 separate branches.

#### ***TL;DR***  
*The moral of the story is that domain knowledge is absolutely crucial for for any data science project. A solid understanding of the features and underlying data is required in order to build a useful model.*  

*Now that there is a deeper understanding of the forecasting domain knowledge, this project's development will be corrected in 2 different ways, through 2 different branches.*  

Stay tuned! :smirk: :wave:

---
