import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

# Descending order to find anomalous data, highest differences - predicts.sort_values("DIFF", ascending=False)

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

    # Temperature Prediction GUI
    # Window
    root = tk.Tk()
    root.title("Temperature Prediction")
    root.geometry("600x600")
    # User enters desired day 
    tk.Label(root, text="Enter the date you would like to predict the temperature for:").place(x=185,y=5)
    # Inputs
    tk.Label(root, text="Year in 0000").place(x=20,y=45)
    tk.Label(root, text="Month in 00").place(x=20,y=85)
    tk.Label(root, text="Day in 00").place(x=20,y=125)
    # Entries
    entry_year = tk.Entry(root, bd=5)
    entry_year.place(x=140,y=45)
    entry_month = tk.Entry(root, bd=5)
    entry_month.place(x=140,y=85)       
    entry_day = tk.Entry(root, bd=5)
    entry_day.place(x=140,y=125)   
    # Button
    tk.Button(root, text="Enter", command=root.quit, height=1, width=10, bd=5).place(x=185, y=165)
    # Call
    root.mainloop()
    year = entry_year.get()
    month = entry_month.get()
    day = entry_day.get()
    root.destroy()
    
    # Date formatting
    date = "{0}-{1}-{2}".format(year, month, day)
    
    try:
        tk.messagebox.showinfo('Temperature', 'The temperature was predicted to be: {0}\n The temperature was actually {1}'.format(get_prediction(date), get_temp(date)))
    except KeyError:
        tk.messagebox.showinfo('Temperature','Data not found.')    

def check(event):
    # Checks menu option selected and returns
    event = event.get()
    return

def main():
    """
    the main function to be ran when the program runs
    """
    running = True 

    while running:
        # Menu GUI 
        menu = tk.Tk()
        menu.title("Menu")
        menu.geometry("600x600")
        # Options
        options = [
                'Predict Temperature of a day',
                'View accuracy of the weather prediction model',
                'View Accuracy Diagram',
                'Quit'
        ]
        # Selection
        selected = tk.StringVar(menu)
        selected.set("Select an option")
        tk.Label(menu, text="Menu").pack()
        drop = tk.OptionMenu(menu, selected, *options, command=check(selected))
        drop.pack(pady = 50)
        # Button
        tk.Button(menu, text="Submit", command=menu.quit).pack(pady=1)
        # Call
        menu.mainloop()
        op = selected.get()
        menu.destroy()
        
        if op == 'Predict Temperature of a day':
            # 1 - Predict Temperature of a day
            predict_weather()

        elif op == 'View accuracy of the weather prediction model':
            # 2 - View accuracy of the weather prediction model
            # Average of difference - Still around 3 degrees off on average 
            degree_avg = str(predicts["DIFF"].mean())
            # NEW WINDOW
            tk.messagebox.showinfo('Mean Difference',"The model's mean difference is {0}".format(degree_avg))    
        
        elif op == 'View Accuracy Diagram':
            # 3 - View Accuracy Diagram
            # Error overview
            # ADD BUTTON TO RETURN 
            # figure1 = plt.Figure(figsize=(6, 5), dpi=100)
            # ax1 = figure1.add_subplot(111)
            # predicts["DIFF"].round().value_counts().sort_index().plot()
            pass
                    
        elif op == 'Quit':
            # 4 - Quit
            # Confirmation berfore quitting
            root = tk.Tk()
            root.withdraw()
            answer = tk.messagebox.askyesno(title='Confirmation',message='Are you sure that you want to quit?')
            if answer:
                root = tk.Tk()
                root.withdraw()
                tk.messagebox.showinfo('Registration', 'You have quit the system.')
                running = False
            else:
                tk.messagebox.showinfo('Return', 'You will now return to the application screen')

if __name__ == '__main__':
    main()

            

