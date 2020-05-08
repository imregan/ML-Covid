import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime

from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm  # mostly for p values

# Alternative option to NY enriched
def johns_hopkins_us_daily_reports():
    files = glob.glob("data/csse_daily_reports*.csv")
    li = []
    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.drop(['Lat', 'Long_', 'ISO3', 'UID', 'FIPS',
                'Country_Region'], axis=1, inplace=True)
    return frame


# join datasets on covid, demographics, and population
def combine_datasets(write_csv=False):
    # put together csv files5
    covid = pd.read_csv("enricheddata/covid19_us_county.csv")
    demo = pd.read_csv("enricheddata/us_county_demographics.csv")
    demo['county'] = demo['county'].apply(lambda x: x.rsplit(' ', 1)[0])  # remove 'County' from string for join
    pop = pd.read_csv("enricheddata/us_county_pop_and_shps.csv")
    
    # very fun
    df = pd.merge(covid, demo, how='left', left_on=['state_fips', 'county_fips', 'state', 'county'], right_on=['state_fips', 'county_fips', 'state', 'county'])
    df = pd.merge(df, pop, how='left', left_on=['state', 'county', 'fips'], right_on=['state', 'county', 'fips'])
    
    df.reset_index(drop=True, inplace=True)  # get rid of index column
    df.dropna(inplace=True)

    if write_csv:
        df.to_csv("covid_enriched.csv")

    return df

# Note that 'colnames' is a list of columns that we wish
# to add as a variable, with values from 'days_back' days
def add_recurrent_cols(df, colnames, days_back):
    # add recurrent variables
    # initialize columns
    for col in colnames:
        newcol = col + "-" + str(days_back)
        df[newcol] = np.zeros_like(df[col])

    # fill in recurrent info by county
    for county in df.county_fips.unique():
        idxs = df.index[df['county_fips'] == county].tolist()

        if len(idxs) < days_back:  # cant look back if not enough entries for county
            break

        # this actually works now I think
        for i in range(days_back, len(idxs)):
            for col in colnames:
                newcol = col + "-" + str(days_back)
                pastval = df.loc[idxs[i-1], col]
                df.loc[idxs[i], newcol] = pastval

    return df

# Trying stuff out with NY data
df = combine_datasets(write_csv=False)
NY = df[df['state'] == 'New York']
NY = NY[['cases', 'county_fips', 'new_day_cases_per_capita_100k', 'pop_per_sq_mile_2010', 'AGE_15TO24','AGE_25TO34','AGE_35TO44','AGE_45TO54','AGE_55TO64','AGE_65TO74','AGE_75TO84','AGE_84PLUS']]
for i in range(1, 10):
    NY = add_recurrent_cols(NY, ['cases', 'new_day_cases_per_capita_100k'], i).dropna()
recurr_params = [[f'cases-{x}', f'new_day_cases_per_capita_100k-{x}'] for x in range(1,10)]
params = ['new_day_cases_per_capita_100k'] + [item for sublist in recurr_params for item in sublist]

# set up response/predictors
X = NY.drop(['new_day_cases_per_capita_100k', 'county_fips', 'cases'], axis=1)
y = NY[['new_day_cases_per_capita_100k']]

# try logistic regression on all
lr = sm.OLS(y, sm.add_constant(X)).fit()
print(lr.summary())

# try feature selection + boosting boosting
boost = GradientBoostingRegressor()
selector = RFE(boost).fit(X,y)
y_pred = cross_val_predict(selector, X, y, cv=10, n_jobs=-1)
r2 = r2_score(y, y_pred)
print(f"Boosting r2 score: {r2}")
