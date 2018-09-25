# weather_db

Forecasting Canada's weather by analyzing top forecasters's data and improving upon their prediction errors. What sets this weather forecasting engine apart from its competitors is that doesn't use meteorology at all — no fancy weather models, no radar projections. Pure data is its only knowledge and ML is the secret sauce. 


###### Primary Goal:
* Predict tomorrow's high with as much accuracy as possible. 
 * A real success would be to have the lowest RMSE between predicted and recorded highs out of all tracked forecasters.

 
###### Secondary Goals:
* Find out which weather forecaster has the best predictions and under which circumstances.
* Find out which features are the most important to predict the weather without meteorology.
* Find out if rumours of forecaster [wet bias](<https://en.wikipedia.org/wiki/Wet_bias>) are valid.
* Collect a dataset of historical forecasts across the country for whatever new purpose the open source community might find


---

Please understand this project is still work in progress and is very much **incomplete**.  
See [Installation](#Installation) instructions to check out the source code on your machine and maybe even help me out.

---



## Challenges

### Data Drift
Given that Canada is a country with 4 seasons, historical daily averages are constantly on the move from one day/week/month to the next. It's unreasonable to expect a machine learning model trained on January data to perform just as well in July. This boils down to the well-known issue of data drift, where the continuous mutation of feature characteristics results in unpredictability of the target variable. In this application, we are always trying to predict tomorrow's wether, which means we are trying to predict an edge case for which the model has been poorly trained on. One way to solve this would be to train the model on a period of time which equally straddles the forecast date. As we've been collecting data for less than 1 year however, this is solution is not yet possible. 

### Micro-Climates
On any given day, the weather across Canada's vast territory tends to vary immensely depending on geographic location. If this is not controlled for, the ML model will be unsuccessful. We solve this issue by pulling in each location's latitude and longitude from Google's Geocoding API. This means the ML model can cater its predictions based on geographic location.

### Geographic Sparsity
Given that Canada is sparsely populated, regions included in weather forecasts tend to be clustered and not very well distributed. This challenge is mostly resolved by using the latitude and longitude features, as described above, but there is still an issue with the fact that we have very little data on the North Territories. It might be worthwhile to exclude every datapoint above a certain latitude threshold — even if it means losing the functionality to predict in those regions.

### Timing of Data Availability
Since we are trying to predict next-day weather with the data we have during the current day, it behooves us to receive data in a timely manner. A major component of being able to forecast the weather lies in the concept of [weather persistence](<https://en.wikipedia.org/wiki/Weather_forecasting#Persistence>). Basically this means that the weather tends to maintain itself over short periods of time. The short-term past is a good predictor of the near future. Therefore, gathering current day conditions as a feature set would surely be of massive benefit for the ML model's performance.  

Unfortunately, both of our tracked forecasters (Environment Canada and The Weather Network) do not offer current day reports in terms of recorded highs and recorded lows. This means we can only use yesterday's conditions for the "persistence" component of our feature set. This data is 48 hours removed from our target prediction's occurrence, which is not very useful. 

This realization leads us to 2 possible courses of actions — both of which will be explored in the future.

1. Switch the forecasting service from predicting tomorrow's weather to predicting today's weather.
    * To achieve this, the ETL process would have to happen ASAP in the morning. However, we would need as many data points as possible from our tracked forecasters. See [this notebook](<https://github.com/confirmationbias616/weather_db/blob/master/Notebooks/History_ETL_Analysis.ipynb>) to see the analysis on what time of day we could potentially start predicting and using which forecaster's data points.
     * This service would be less useful but still worthwhile to certain niche customers (such as scientists, construction consultants, festival organizers) 
2. Find a weather forecaster who tracks and reports actual-day highs and lows for a wide range of Canadian locations. 


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
Unpickle appropriate model generated in previous step and use it to predict

#### Report
Provide some insight into how the model predicted the weather for yesterday.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Make sure you are running Python3 and have installed all required packages using `pip install -r requirements.txt`

### <a name="Installation"></a>Installation
* Clone this repo to your computer.
* Get into the folder using ```cd weather_db```.
* Run ```mkdir Data Gym Predictions```.
* Download contents from each of the following dropbox links and copy them to your local respective newly created directories
    * [Data](https://www.dropbox.com/sh/wh9feldwhj3wagu/AADdnMciGo5Jk8WcIk3m5uTNa?dl=0)
    * [Gym](https://www.dropbox.com/sh/cai718eeplvb8fi/AAAF1Nnz5w098HDgIH0WoWg3a?dl=0)
    * [Predictions](https://www.dropbox.com/sh/y0v5l20oyocyb7v/AADDkzurHZikFe9sfu5jYIq1a?dl=0)

### Schedule
Currently, I run 2 scripts on a daily schedule, using launchd through a great little app called [LaunchControl](<http://www.soma-zone.com/LaunchControl/>):
* `Day.py` at 5:10 PM
* `Evening.py` at 10:10 PM
If you want the same results as me, you should do the same.


## To-Do
Here is a very rough to-do list, which is constantly evolving.

* Try filtering out all cities above certain latitude (both at the training level and testing level). Could also try simply filtering out all predictions that are 2 or more degrees off of any of the weather providers. (But that wouldn’t feel very good).

* Make a script to backfill missing values for history_db at the end of every month.

* Get set up with YELLOWBRICK and explore!

* Tweak ETL a bit to make sure Ottawa is being tracked

* Set up creation of daily data viz to show recent history of Ottawa prediction performances by TWN vs EC vs ML vs actual... play around with Klipfolio API??

* Is there anything we can do with quantile regression?! Reduce the amount of extreme predictions?

* Create a few datasets with different lengths of trailing dates to be used in larger grid search loop of hyperparameterization
    * random search?
    * grid search with starting point from hyperparameters of yesterday’s model, +\- a certain range a percentage variation.
    * Try out different time horizons using tweaked model and use prediction of yesterday’s high as the 

* Create a new feature, ‘days_ago’, on the fly (inside the pipeline instance) right before every ML training session. 

* Smooth out the fluctuations on the averages. There’s no reason for the 2 degree jump in average from one day to the next. Completely throws off the model. 

* Plot latitude vs longitude to get a feel for location distribution. Overlay on map?

* In the post-mortem script, identify how many locations were better served by TWN than ML. Same for EC. So many stats can be pulled from this. Can we even use it as a metric in the outer hyperparameterization loop?!

* Things to clean up:
    * Clean any line that verbosely outputs to the log or posts a warning to the log
    * Remove references to dB’s? Or just explain in readme that it’s because the project will be switching to a relational database eventually. 
    * PEP8

* Set up a way of travelling back in time so that we can do heavy experimentation on hyperparameterization and not have to wait a day for testing the results because we don’t have the data yet. It would be wicked sweet to roll through the history in one loop and collect results for a certain set of hyperparameters 

* How about we use a hybrid of models for the prediction? 1 recent country wide + 1 long term localized? 1 recent weighted heavy + 1 long term weighted less?

* Turn (almost) all .py flies into functions. Then, current date could be passed in as a variable. This would allow us to time travel

* make ETL functions more efficient by merging data frame into master CSV only one at the end (not at every loop!)

* need to modify code to prepare for snow accumulation instead of rain accumulation, which is usually reported in cm (vs mm for rain!).
