from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


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


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size= .25)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

print(model.summary())

model.fit(X_train, y_train, epochs=40, batch_size=64, verbose=1)  # ,validation_data=(X_test,y_test)
train_predict = model.predict(X_train)
print("Train error", mean_squared_error(y_train, train_predict))
# test_predict = model.predict(X_test_orignal)

output = pd.DataFrame(model.predict(X_test_orignal), columns=['target'])
output.to_excel('Output.xlsx')



