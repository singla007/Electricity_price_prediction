import pandas as pd
from sklearn.metrics import mean_squared_error

train_df = pd.read_excel("Train.xlsx")
test = pd.read_excel("Test.xlsx")
y_train = train_df['target']
train_df = train_df.drop(['target'], axis=1)
size_of_train = train_df.shape[0]
train = train_df.append(test, ignore_index=True)
print(train)

print(train.describe())

print(train.isnull().sum().sort_values(ascending=False))

train = pd.concat([train, pd.get_dummies(train['Hour'], prefix='Hour', drop_first=True)], axis=1)
train = pd.concat([train, pd.get_dummies(train['Weekday'], prefix='Weekday', drop_first=True)], axis=1)
train = pd.concat([train, pd.get_dummies(train['Is Working Day'], prefix='Is Working Day', drop_first=True)], axis=1)
train.drop(['Is Working Day'], axis=1, inplace=True)
train.drop(['Hour'], axis=1, inplace=True)
train.drop(['Weekday'], axis=1, inplace=True)
train.drop(['Date'], axis=1, inplace=True)

X_train = train.iloc[:size_of_train, :]
X_test_orignal = train.iloc[size_of_train:, :]


# from flaml import AutoML
# automl = AutoML()
# automl_settings = {
#     "time_budget": 10,  # in seconds
#     "metric": 'r2',
#     "task": 'regression',
#     # "log_file_name": "test/iris.log",
# }
#
# # Train with labeled input data
# automl.fit(X_train=X_train, y_train=y_train, verbose=3, estimator_list=["xgboost"],
#                         **automl_settings)
# # output = pd.DataFrame(automl.predict(X_test_orignal), columns=['target'])
# # output.to_excel('Output.xlsx')
#
# y_pred = automl.predict(X_train)
# print("Train error", mean_squared_error(y_train, y_pred))


from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size= .25)
reg = TPOTRegressor(verbosity=2, population_size=50, generations=10, random_state=35)

reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

#save the model in top_boston.py
reg.export('top_boston.py')


