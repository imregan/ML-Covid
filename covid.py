import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

import datetime

from preprocessing import set_column_datetimes

# Quick try at johns hopkins data parsing global time series data for specific country
# kinda messy
# returns dataframe with dates, cases, and log cases of specific country
def johns_hopkins_global_timeseries(country='US'):
    # read data and fix datetime format
    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    set_column_datetimes(df)

    # messy setup
    df = df.drop(['Province/State', 'Lat', 'Long'], axis=1).T
    df = df.rename(columns=df.iloc[0]).drop(df.index[0])
    df = df.reset_index().rename(columns={'index':'date'})
    df = df[[country, 'date']]
    df = df.rename(columns={country:'cases'})

    # add log cases
    m = df['cases'].values.astype('float64')
    df['log_cases'] = np.log10(m, out=np.zeros_like(m), where=(m!=0))

    print(f"Case data for country: {country}")
    return df


# parsing data from our world in data
# much nicer formatting by default
def owid(country="United States"):
    df = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv").fillna(0)
    df.drop(["iso_code"], axis=1, inplace=True)
    
    exp_data = ["total_cases", "new_cases"]  # stuff we want to map log of instead
    for col in exp_data:
        logstring = "log_" + str(col)
        m = df[col].values.astype('float64')
        df[logstring] = np.log10(m, out=np.zeros_like(m), where=(m>0))
        
    if country is not None:
        df = df[df["location"] == country]
    
    return df

# copy over specified columns from days back
def owid_recurrent(df, columns, days_back):
    # initialize columns
    for col in columns:
        newcol = col + "-" + str(days_back)
        df[newcol] = np.zeros_like(df[col])

    # fill recurrent info for each country
    for country in df.location.unique():
        idxs = df.index[df['location'] == country].tolist()
        start, stop = idxs[0], idxs[-1]
        for col in columns:
            newcol = col + "-" + str(days_back)   # maybe rename this later
            for i in range(start+days_back, stop+1):
                df.loc[i, newcol] = df.loc[i-days_back, col]


# sanity check with recurrent helper
df = owid()
owid_recurrent(df, ['total_cases'], 7)

X = df[['total_cases','total_cases-7', 'total_tests', 'new_tests']]
y = df[['new_cases']]

lr = sm.OLS(y, X).fit()
print(lr.summary())
