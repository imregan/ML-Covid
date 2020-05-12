import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

plt.style.use('seaborn-whitegrid')
import datetime

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.interpolate import CubicSpline
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
    demo['county'] = demo['county'].apply(lambda x: x.rsplit(
        ' ', 1)[0])  # remove 'County' from string for join
    pop = pd.read_csv("enricheddata/us_county_pop_and_shps.csv")

    # very fun
    df = pd.merge(covid, demo, how='left', left_on=['state_fips', 'county_fips', 'state', 'county'], right_on=[
        'state_fips', 'county_fips', 'state', 'county'])
    df = pd.merge(df, pop, how='left', left_on=[
        'state', 'county', 'fips'], right_on=['state', 'county', 'fips'])

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
                pastval = df.loc[idxs[i - days_back], col]
                df.loc[idxs[i], newcol] = pastval

    return df


# plot predictions for models below
def plot_prediction(X, y_test, y_pred, title):
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Cases")
    days = [i for i in range(X.shape[0])]
    plt.scatter(days, y_test)
    plt.plot(days, y_pred, color='red')
    plt.show()


# Ttying some simple models with data from one county
df = combine_datasets(write_csv=False)
print(df.county.mode, df.county_fips.mode)  # try county with most data points
X = df[(df['county'] == 'Suffolk') & (df['state'] == 'Massachusetts')].sort_values(
    by='date')  # need to filter by both state and county because there are duplicates

# filter parameters
X = X[['cases', 'cases_per_capita_100k', 'county_fips', 'new_day_cases_per_capita_100k', 'pop_per_sq_mile_2010',
       'AGE_15TO24',
       'AGE_25TO34', 'AGE_35TO44', 'AGE_45TO54', 'AGE_55TO64', 'AGE_65TO74', 'AGE_75TO84', 'AGE_84PLUS', 'new_day_cases']]
# log scale cases
# X[['cases']] = np.log(X[['cases']])

# set up recurrent structure
for i in range(1, 10):
    X = add_recurrent_cols(
        X, ['cases'], i).dropna()
recurr_params = [
    [f'cases-{x}'] for x in range(3, 8)]
# params = ['pop_per_sq_mile_2010', 'AGE_15TO24',
#           'AGE_25TO34', 'AGE_35TO44', 'AGE_45TO54', 'AGE_55TO64', 'AGE_65TO74', 'AGE_75TO84', 'AGE_84PLUS'] + \
#     [item for sublist in recurr_params for item in sublist]
params = ['pop_per_sq_mile_2010', 'AGE_15TO24',
          'AGE_25TO34', 'AGE_35TO44', 'AGE_45TO54', 'AGE_55TO64', 'AGE_65TO74', 'AGE_75TO84', 'AGE_84PLUS',
          'cases-1', 'cases-4', 'cases-8']
# sanity check
map(lambda s: s.rstrip(), params)

# set up response/predictors (testing by removing last 10 days of data)
X_train = X[params].drop(X[params].tail(10).index)
X_test = X[params]
y_train = X[['cases']].drop(X[params].tail(10).index).values.ravel()
y_test = X[['cases']].values.ravel()


# plot some points to see this data yo
def plot_county(title, metric):
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Cases")
    days = [i for i in range(X.shape[0])]
    print(days)
    yvals = X[[metric]].values.ravel()
    plt.scatter(days, yvals)
    plt.show()


plot_county('New Cases per Day in Suffolk County MA', 'ca')

# try linear regression on all
# lr = sm.OLS(y_train, sm.add_constant(X_train)).fit()
# y_pred = lr.predict(sm.add_constant(X_test))
# error = mean_absolute_error(y_test, y_pred)
# print(f"Linear error: {error}")
#
# print(lr.summary())
#
# # plot LR
# # plot_prediction(X, y_test, y_pred, "Linear Regression")
#
# # lasso
# lasso = Lasso(alpha=1).fit(X_train, y_train)
# y_pred = lasso.predict(X_test)
# error = mean_absolute_error(y_test, y_pred)
# print(f"LASSO error: {error}")
#
# # plot lasso
# plot_prediction(X, y_test, y_pred, "Lasso Regression")
#
# # elastic net
# en = ElasticNet(alpha=1, random_state=0).fit(X_train, y_train)
# y_pred = en.predict(X_test.values)
# error = mean_absolute_error(y_test, y_pred)
# print(f"Elastic net error: {error}")
# #
# # plot elastic net
# plot_prediction(X, y_test, y_pred, "Elastic Net Regression")
#
# # random forests -- strange end behavior -- overfitting?
# forest = RandomForestRegressor(
#     n_estimators=25, random_state=0, n_jobs=-1, max_features='log2').fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# error = mean_absolute_error(y_test, y_pred)
# print(f"Random forest error: {error}")
# #
# # plot forests
# plot_prediction(X, y_test, y_pred, "Random Forest Regression")

# # feature selection with LASSO  # TODO finish
# sfm = SelectFromModel(Lasso(alpha=1), threshold=0.2)
# sfm.fit(X_train, y_train)
# n_features = sfm.transform(X_train).shape[1]
# print(X_train)
# #

# selection for number of cases
