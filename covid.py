import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import datetime

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
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
                pastval = df.loc[idxs[i-days_back], col]
                df.loc[idxs[i], newcol] = pastval
    return df


# place recurrent variables
def place_recurrent_variables(X, response_var='cases', days_back_range=[3, 4, 5, 6, 7]):
    for lag in days_back_range:
        X = add_recurrent_cols(X, [response_var], lag).dropna()
    recurr_params = [[f'{response_var}-{x}'] for x in days_back_range]
    params = ['pop_per_sq_mile_2010', 'cases', 'AGE_15TO24',
              'AGE_25TO34', 'AGE_35TO44', 'AGE_45TO54', 'AGE_55TO64', 'AGE_65TO74', 'AGE_75TO84', 'AGE_84PLUS'] + \
        [item for sublist in recurr_params for item in sublist]
    return X[params]


# plot predictions for models below
def plot_prediction(X, y_test, y_pred, title):
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Cases")
    days = [i for i in range(X.shape[0])]
    plt.scatter(days, np.exp(y_test))
    plt.plot(days, np.exp(y_pred), color='red')
    plt.show()


# set up testing for state for training, county for testing
def state_testing(df, state_str, county_str, response_var='cases', days_back_range=[3, 4, 5, 6, 7]):
    state = df[df['state'] == state_str]
    # state = state.drop(state[state['cases'] < 10].index) # not working
    county = state[state['county'] == county_str]
    train = state.drop(county.index)
    if response_var == 'cases':   # log scale for cases
        train[['cases']], county[['cases']] = np.log(
            train[['cases']]), np.log(county[['cases']])
    train = place_recurrent_variables(train, response_var, days_back_range)
    test = place_recurrent_variables(county, response_var, days_back_range)
    X_train, X_test, y_train, y_test = train.drop([response_var], axis=1), test.drop(
        [response_var], axis=1), train[[response_var]], test[[response_var]]
    return X_train, X_test, y_train, y_test


# filter out unimportant predictors (not really used anymore)
def filter_columns(X):
    X = X[['pop_per_sq_mile_2010', 'AGE_15TO24',
           'AGE_25TO34', 'AGE_35TO44', 'AGE_45TO54', 'AGE_55TO64', 'AGE_65TO74', 'AGE_75TO84', 'AGE_84PLUS']]
    return X


# set up testing for specific county with n test days
def county_testing(df, state_str, county_str, n_test_days=10, response_var='cases', days_back_range=[3, 4, 5, 6, 7]):
    state = df[df['state'] == state_str]
    county = state[state['county'] == county_str]
    if response_var == 'cases':
        county[response_var] = np.log(county[response_var])
    data = place_recurrent_variables(county, response_var, days_back_range)
    X = data.drop([response_var], axis=1)
    y = data[[response_var]]
    X_train, X_test, y_train, y_test = X.drop(
        X.tail(n_test_days).index), X, y.drop(y.tail(n_test_days).index).values.ravel(), y.values.ravel()
    return X_train, X_test, y_train, y_test


# Load datasets
df = combine_datasets(write_csv=False)
#X_train, X_test, y_train, y_test = state_testing(df, 'Washington', 'Snohomish')
X_train, X_test, y_train, y_test = county_testing(
    df, 'Washington', 'Snohomish', n_test_days=10)


# try linear regression on all
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
print(f"Linear error: {error}")

# plot LR
plot_prediction(X_test, y_test, y_pred, "Linear Regression")

# lasso
lasso = Lasso(alpha=1, max_iter=50000).fit(X_train, y_train)
y_pred = lasso.predict(X_test)
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
print(f"LASSO error: {error}")

# plot lasso
plot_prediction(X_test, y_test, y_pred, "Lasso Regression")

# elastic net
en = ElasticNet(alpha=1, max_iter=50000, random_state=0).fit(X_train, y_train)
y_pred = en.predict(X_test.values)
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
print(f"Elastic net error: {error}")

# plot elastic net
plot_prediction(X_test, y_test, y_pred, "Elastic Net Regression")

# random forests -- strange end behavior -- overfitting?
forest = RandomForestRegressor(
    n_estimators=50, random_state=0, n_jobs=-1).fit(X_train, y_train)
y_pred = forest.predict(X_test)
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
print(f"Random forest error: {error}")

# plot forests
plot_prediction(X_test, y_test, y_pred, "Random Forest Regression")

# feature selection with linear estimator and cross validation
selector = RFECV(LinearRegression()).fit(X_test, y_test)
cols = selector.get_support(indices=True)
features = X_train.iloc[:, cols]

print(features.head())
