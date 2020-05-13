import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import datetime

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression, SelectFromModel
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from scipy.interpolate import CubicSpline
import statsmodels.api as sm  # mostly for p values


# join datasets on covid, demographics, and population
def combine_datasets(write_csv=False):
    # put together csv files5
    covid = pd.read_csv('enricheddata/covid19_us_county.csv')
    demo = pd.read_csv('enricheddata/us_county_demographics.csv')
    demo['county'] = demo['county'].apply(lambda x: x.rsplit(
        ' ', 1)[0])  # remove 'County' from string for join
    pop = pd.read_csv('enricheddata/us_county_pop_and_shps.csv')

    # very fun
    df = pd.merge(covid, demo, how='left', left_on=['state_fips', 'county_fips', 'state', 'county'], right_on=[
                  'state_fips', 'county_fips', 'state', 'county'])
    df = pd.merge(df, pop, how='left', left_on=[
                  'state', 'county', 'fips'], right_on=['state', 'county', 'fips'])

    df.reset_index(drop=True, inplace=True)  # get rid of index column
    df.dropna(inplace=True)

    if write_csv:
        df.to_csv('covid_enriched.csv')

    return df


# Note that 'colnames' is a list of columns that we wish
# to add as a variable, with values from 'days_back' days
def add_recurrent_cols(df, colnames, days_back):
    # add recurrent variables
    # initialize columns
    for col in colnames:
        newcol = col + '-' + str(days_back)
        df[newcol] = np.zeros_like(df[col])

    # fill in recurrent info by county
    for county in df.county_fips.unique():
        idxs = df.index[df['county_fips'] == county].tolist()

        if len(idxs) < days_back:  # cant look back if not enough entries for county
            break

        # this actually works now I think
        for i in range(days_back, len(idxs)):
            for col in colnames:
                newcol = col + '-' + str(days_back)
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
    plt.xlabel('Day')
    plt.ylabel('Cases')
    days = [i for i in range(X.shape[0])]
    plt.scatter(days, np.exp(y_test), s=20, alpha=0.7)
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


# diagnostic plot
def resid_fitted_plot(X, y, graph_title):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    # model values
    model_fitted_y = model.fittedvalues
    # model residuals
    model_residuals = model.resid
    # normalized residuals
    model_norm_residuals = model.get_influence().resid_studentized_internal

    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, y,
                                      lowess=True,
                                      scatter_kws={'alpha': 0.5},
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')

    plt.title(graph_title)
    plt.show()


# set up testing for specific county with n test days
def county_testing(df, state_str, county_str, n_test_days=7, response_var='cases', days_back_range=[1, 2, 3, 6, 7]):
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


# exp transform for real case error
def mean_case_error(y, y_pred):
    cases = np.exp(y)
    pred = np.exp(y_pred)
    return mean_absolute_error(cases, pred)


# model parameter tuning
def parameter_tuning(estimator, X, y, params, graph_title):
    scorer = make_scorer(mean_case_error, greater_is_better=False)
    grid = GridSearchCV(estimator, params,
                        cv=TimeSeriesSplit(n_splits=5), scoring=scorer, n_jobs=-1).fit(X, y)
    print(f'Best estimator for model {graph_title}: \n {grid.best_estimator_}')
    grid_results = grid.cv_results_
    mean_score = grid_results['mean_test_score']

    if len(params) == 1:  # lasso
        key = list(params.keys())[0]
        plt.plot(params[key], mean_score)
        plt.xlabel(key)
        plt.ylabel('Mean Absolute Error (Cases)')
        plt.title(graph_title)
        plt.show()

    else:
        key1, key2 = list(params.keys())[0], list(params.keys())[1]
        params1, params2, = params[key1], params[key2]
        mean_score = np.array(mean_score).reshape(len(params2), len(params1))
        for idx, val in enumerate(params2):
            plt.plot(params1, mean_score[idx, :],
                     label=key2 + ': ' + str(val))
            plt.xlabel(key1)
            plt.ylabel('Mean Absolute Error (Cases)')
            plt.legend(loc='best')
            plt.title(graph_title)
            plt.show()


# Load datasets
df = combine_datasets(write_csv=False)
# X_train, X_test, y_train, y_test = state_testing(df, 'Washington', 'Snohomish')
X_train, X_test, y_train, y_test = county_testing(
    df, 'Washington', 'Snohomish', n_test_days=1)

# comment out to give all params
final_params = ['cases-1', 'cases-7']
X_train, X_test = X_train[final_params], X_test[final_params]


# resid_fitted_plot(X_train, y_train, 'fitted residuals cases-17')  # plot residuals for current params

# try linear regression on all
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
error = mean_absolute_error(
    np.exp(y_test), np.exp(y_pred))  # mean absolute error
day_pred_error = np.exp(y_pred[-1]) - np.exp(y_test[-1])
print(f'Day prediction error {day_pred_error}, Percent Error {100*(day_pred_error/np.exp(y_test[-1]))}')
print(f'Mean Linear error: {error}')

# plot LR
plot_prediction(X_test, y_test, y_pred, 'Linear Regression - Snohomish')

# lasso
#params = {'alpha': np.linspace(0, 1, 101)}
# parameter_tuning(Lasso(max_iter=50000, random_state=0),
#                 X_train, y_train, params, 'Lasso CV')
lasso = Lasso(alpha=0.2, max_iter=50000, random_state=0).fit(X_train, y_train)
y_pred = lasso.predict(X_test)
day_pred_error = np.exp(y_test[-1]) - np.exp(y_pred[-1])
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
print(f'Day prediction error {day_pred_error}, Percent Error {100*(day_pred_error/np.exp(y_test[-1]))}')
print(f'LASSO mean error: {error}')

# plot lasso
plot_prediction(X_test, y_test, y_pred, 'Lasso Regression - Snohomish')

# elastic net
# params = {'alpha': np.linspace(0, 1, 101), 'l1_ratio': np.linspace(0, 1, 11)} # real testing
# params = {'alpha': np.linspace(0, 1, 101), 'l1_ratio': [
#    0.1, 0.5, 0.9]}  # easier to graph
# parameter_tuning(ElasticNet(max_iter=50000, random_state=0), X_train,
#                 y_train, params, 'Elastic Net CV')
en = ElasticNet(alpha=0.02, l1_ratio=0.1, max_iter=500000,
                random_state=0).fit(X_train, y_train)
y_pred = en.predict(X_test.values)
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
day_pred_error = np.exp(y_test[-1]) - np.exp(y_pred[-1])
print(f'Day prediction error {day_pred_error}, Percent Error {100*(day_pred_error/np.exp(y_test[-1]))}')
print(f'Elastic net mean error: {error}')

# plot elastic net
plot_prediction(X_test, y_test, y_pred, 'Elastic Net Regression - Snohomish')

# random forests -- strange end behavior -- overfitting?
# params = {'n_estimators': [i for i in range(
#    10, 501, 5)], 'max_features': ['auto', 'sqrt']}
# parameter_tuning(RandomForestRegressor(criterion='mae'), X_train,
#                 y_train, params, 'Random Forests')
forest = RandomForestRegressor(
    n_estimators=10, random_state=0, n_jobs=-1).fit(X_train, y_train)
y_pred = forest.predict(X_test)
error = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
day_pred_error = np.exp(y_test[-1]) - np.exp(y_pred[-1])
print(f'Day prediction error {day_pred_error}, Percent Error {100*(day_pred_error/np.exp(y_test[-1]))}')
print(f'Random forest mean error: {error}')

# plot forests
plot_prediction(X_test, y_test, y_pred, 'Random Forest Regression - Snohomish')

# feature selection with LinearRegression and cross validation
selector = RFECV(LinearRegression(), cv=TimeSeriesSplit(
    n_splits=5)).fit(X_test, y_test)
cols = selector.get_support(indices=True)
features = X_train.iloc[:, cols]

# cases-1 and cases-7 on snohomish
print(f'Chosen Features on Snohomish Dataset: {features.columns}')

# # repeat with Elastic Net as a sanity check
# selector = RFECV(ElasticNet(), cv=TimeSeriesSplit(
#     n_splits=5)).fit(X_test, y_test)
# cols = selector.get_support(indices=True)
# features = X_train.iloc[:, cols]

# # should show that cases-1 and cases-7 are the best
# print(f'Chosen Features on Snohomish Dataset: {features.columns}')
