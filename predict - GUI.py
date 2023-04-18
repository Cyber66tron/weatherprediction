# Machine learning 
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
# GUI
import tkinter as tk
from tkinter import messagebox
# Graph
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Icon
from urllib.request import urlopen
from PIL import Image, ImageTk
import io 

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

# Returns predicted and actual temperature
def get_prediction(date):
    # Window
    predict_window = tk.Tk()
    predict_window.title("Prediction")
    predict_window.geometry("600x600")

    # Return
    r1 = tk.Button(predict_window, text="Return", command=predict_window.quit, height=1, width=10, bd=5)

    # Prediction
    tk.Label(predict_window, text='The weather was predicted to be: ').pack()
    tk.Label(predict_window,text=f'{predicts["PREDICTION"].loc[date]}').pack()
    tk.Label(predict_window, text='The weather was actually: ').pack()
    tk.Label(predict_window,text=f'{predicts["DATA"].loc[date]}').pack()
            
    r1.pack()

    # Call
    predict_window.mainloop()
    predict_window.destroy()   

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
        get_prediction(date)
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

        # Logo 
        url = "https://lh3.googleusercontent.com/YUR7YwhowSzmUa563Jzz8RW-RYz5MYs_UHWZxIlyUEv8-rpGDfjULacvk57L7Ln__bQcHoP86ae23T4g5lYpI-PITvAnAnnpC1RnFM3MLnZwt5ZOwBijj22mlzVEyUiFXyV0m2cuj4BRx5cMhTEHNH9XnAFKuKb3XtGtqf3mWh2AIOJ8SH8qi32SbGnqLfg_rQwkJ5iIWgtLEbgP690c6jAzO17VZ-LCKQluWnm3uKj1cZqZDxp1OwGjCq4FmT8yLAgJpNNL-YebqhUfu50OG_LE73td3-5YnYWb9qmyzER4SN4-I-sFbGBP4LrFYoURBnJ5rxYnDnZjlASKyHrXPFPuaeyftfnHVckyxRz3Xmw2MxkzN6DlBsnDIDszJn38_b-jleLScqY7X5tLuRdX2I82acujzKEGVOqslJ5InUxeKz3XFWjV9GgIWeqF5MwZ6LS1Xe37_KKy344nPU4GUr7LDnyJdQzmJSjkTiCKPtbzR83PNsquDmy3GPTdTTQEXYb3anrPzsg5VOZDZ6nJSkuJi1YVhT75zB1wRnBVNfDHH9I660oJKkNXCposXzCl2wQqYxAZ6DqW0PvJA7bDDLYk5-AMMzYE_CyYJ6lQDqxFQiZEP7EwQxuGRjSe2T6AJoPHDqoCymzXulYx56WU6grbebfNcpzKwM1ohl4ThTdx735lR1OCyMbJkg9e4ut_AJVX0UAI_G-mXc1DV_DQCVOjVMxTqD0l-jDjhvcSp_KSDBC8LPBQcVqsETCWxB-qdn3B-POCxShqKCEpmQ98YpEBecXURGuz0sBnP-3bjze2i_qSBoNzDsrJoeomocpEtJiirYdgnJ5Bd0cjvufnPhzOwlTvkVnvuyPOf2KilfuINeD13E_d_NoD9tTlkhmtjDXvjbQf2ZF7GTC0Kite6un7TLHulDMySvdMdaMqbEQEdHon2oHDIprlPpEpDIfkkCZf_tNhrt8OS_9D=w25-h34-s-no?authuser=1"
        response = urlopen(url)
        image_bytes = response.read()

        # convert the image bytes to a photo image object
        image = Image.open(io.BytesIO(image_bytes))
        photo = ImageTk.PhotoImage(image)

        # set the photo image object as the icon for the window
        menu.tk.call('wm', 'iconphoto', menu._w, photo)

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

            # Window
            mean_window = tk.Tk()
            mean_window.title("Mean Difference")
            mean_window.geometry("600x600")

            # Return
            r1 = tk.Button(mean_window, text="Return", command=mean_window.quit, height=1, width=10, bd=5)
            # Mean
            tk.Label(mean_window, text="The model's mean difference is: ").pack()
            tk.Label(mean_window,text=f'{degree_avg}').pack()

        
            r1.pack()

            # Call
            mean_window.mainloop()
            mean_window.destroy()   
        
        elif op == 'View Accuracy Diagram':
            # 3 - View Accuracy Diagram
            # Error overview

            # Window
            diagram_window = tk.Tk()
            diagram_window.title("Accuracy Diagram")
            diagram_window.geometry("600x600")

            # Return
            r1 = tk.Button(diagram_window, text="Return", command=diagram_window.quit, height=1, width=10, bd=5)
            # Diagram
            tk.Label(diagram_window, text="Accuracy Diagram: ").pack()
            
            # create a matplotlib figure
            fig = plt.figure(figsize=(7,5))

            # plot the data
            ax = fig.add_subplot(111)
            predicts["DIFF"].round().value_counts().sort_index().plot(ax=ax)

            # set the x-axis label
            ax.set_xlabel("Difference in Degrees")

            # set the y-axis label
            ax.set_ylabel("Number of Days")

            # create a tkinter canvas and embed the matplotlib figure in it
            canvas = FigureCanvasTkAgg(fig, master=diagram_window)
            canvas.draw()
            canvas.get_tk_widget().pack()

            r1.pack()

            # Call
            diagram_window.mainloop()
            diagram_window.destroy()
   
        elif op == 'Quit':
            # 4 - Quit
            # Confirmation berfore quitting
            root = tk.Tk()
            root.withdraw()
            answer = tk.messagebox.askyesno(title='Confirmation',message='Are you sure that you want to quit?')
            if answer:
                root = tk.Tk()
                root.withdraw()
                tk.messagebox.showinfo('Quit', 'You have quit the system.')
                running = False
            else:
                tk.messagebox.showinfo('Return', 'You will now return to the application screen')

if __name__ == '__main__':
    main()

            

