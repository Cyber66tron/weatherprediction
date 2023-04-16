import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge

# CSV file with Birmingham weather data
weather = pd.read_csv("weather.csv", index_col="DATE")

# Percentage of null
# Number of null values/Total number of rows
null_per = weather.apply(pd.isnull).sum()/weather.shape[0]

# Remove columns with null
valid_col = weather.columns[null_per <.05]
weather = weather[valid_col].copy()

weather = weather.dropna()


# Fill missing value filled - no null
weather = weather.ffill()
# weather.apply(pd.isnull).sum() - Check for NaN
weather.apply(lambda x: (x == 9999).sum())


# Object type to date type
weather.index = pd.to_datetime(weather.index)


# Check for data gaps
weather.index.year.value_counts().sort_index()

# Shows next day's tmax
weather["NEXT"] = weather.shift(-1)["TMAX"]


# Row filled in as plenty of data to not scew with dataset
weather = weather.ffill()


# Alpha controls how much coefficents are shrunk for collinearity
rr = Ridge(alpha=.1)



# All columns except the following 
predictors = weather.columns[~weather.columns.isin(["NEXT", "NAME", "STATION"])]

# Time series cross-validation
def cross_val(weather, model, predictors, start=3650, step=90):
    all_p = []
    
    # start with 3650, up to end of data set, advance 90
    for i in range(start, weather.shape[0], step):
        # training set - all rows up to i (current)
        train = weather.iloc[:i,:]
        # test set, next 90 days to make predictions
        test = weather.iloc[i:(i+step),:] 
        
        # Predictors and Prediction
        model.fit(train[predictors], train["NEXT"])
        
        predictions = model.predict(test[predictors])
        # Returns numpy array - so format
        predictions = pd.Series(predictions, index=test.index)
        # Real data combined with predictions
        combined = pd.concat([test["NEXT"], predictions], axis=1)
        combined.columns = ["DATA", "PREDICTION"]
        # Prediction vs Actual data 
        combined["DIFF"] = (combined["PREDICTION"] - combined["DATA"]).abs()
        
        # Predictions to one data frame
        all_p.append(combined)
    return pd.concat(all_p)

# Back-testing
predicts = cross_val(weather, rr, predictors)

# NaN values replaced
predicts = predicts.ffill()

# Improve accuracy -  Average temp past few days and compare current day
def perc_diff(old_val, new_val):
    return (new_val - old_val)/old_val

def compute_avg(weather, horizon, col):
    label = f"ROLLING_{horizon}_{col}"
    
    # Rolling mean takes avg of last rows before current 
    weather[label] = weather[col].rolling(horizon).mean()
    # Percentage diff between current and rolling
    weather[f"{label}_PERC"] = perc_diff(weather[label], weather[col])
    return weather 

rolling_horizons = [5, 15]

for horizon in rolling_horizons:
    for col in ["TMAX", "TMIN"]:
        weather = compute_avg(weather, horizon, col)


# Remove first 15 rows, no previous days to first date
# iloc indexes by number, loc by date - 20weather = weather.iloc[15:, :]
# Fill NaN with 0 for missing values resulting from zero div, precaution
weather = weather.fillna(0)

# Find mean of all months before that month and average
# Only months before to prevent bias of taking future data 
def exp_mean(weather):
    return weather.expanding(1).mean()

# Monthly  averages 
for col in ["TMAX", "TMIN"]:
    # Group values per month and creates new column 
    weather[f"MONTH_AVG_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(exp_mean)
    weather[f"DAY_AVG_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(exp_mean)


# All columns except the following 
predictors = weather.columns[~weather.columns.isin(["NEXT", "NAME", "STATION"])]


# Backtesting
predicts = cross_val(weather, rr, predictors)


# Average of difference - Still around 3 degrees off on average 
predicts["DIFF"].mean()


# Descending order to find anomalous data, highest differences - predicts.sort_values("DIFF", ascending=False)
# Error overview - predicts["DIFF"].round().value_counts().sort_index()

# Save model for future use
joblib.dump(weather, 'df_weather.pkl')

# Returns predicted temperature
def get_prediction(date):
    prediction_day = predicts["PREDICTION"].loc[date]
    return prediction_day


# Returns actual temperature 
def get_temp(date):
    temp_day = predicts["DATA"].loc[date]
    return temp_day



from datetime import datetime, timedelta 

def predict_weather():
    weather2 = joblib.load('df_weather.pkl')
    
    # User enters desired day 
    print("Enter the date you would like to predict the temperature for:")
    year = input("Year in 0000:")
    month = input("Month in 00:")
    day = input("Day in 00:")
    
    # Date formatting
    date = "{0}-{1}-{2}".format(year, month, day)
    print("On the day:",date)
    
    try:
        print("The temperature was predicted to be: {0}".format(get_prediction(date)))
        print("The temperature was actually: {0}".format(get_temp(date)))
    except KeyError:
        print("Data not found.")
    
predict_weather()

