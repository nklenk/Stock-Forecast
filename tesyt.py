#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:23:47 2018

@author: neilklenk
"""


#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_data = dataset_train.iloc[:, 1:2].values    # .values turns series obj into an array

# Feature scaling - maxmin average vs standard scaling w/ distribution
from sklearn.preprocessing import MinMaxScaler
# scale all values between 0 and 1
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_data)

# creating a data structure with 60 timesteps and one output
num_timesteps = 60
X_train = []
y_train = []
# For each step, append the previous 60 steps to the X_train to be used as predictors, 
# and append that value to the y_train to be used in validation
for i in range(num_timesteps, training_data.shape[0]):
    X_train.append(training_set_scaled[i-num_timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.asarray(X_train), np.asarray(y_train)

# reshape training data to be compatable with LSTM layer
# 3rd dimension is the number of features(indicators)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Keras packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Initialize model
# Regressor because we are looking for a number
regressor = Sequential()

# Adding first LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding thrid layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding the 4th and final layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the model
# rmsprop is common for RNNs, but in this case adam works better
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

## Making Predictions and visualizing the results

# Getting the true stock prices for 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# Need to conctentat training set onto test set, but must scale test set in the same way we scaled the train set 
# to make a prediction on instance t, you need [t-60:t-1] previous instances
# concatenate origional train and test sets
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# take the last day, and previous 60 days as out test 
inputs = dataset_total[len(dataset_total) - (len(dataset_test) + num_timesteps):].values #numpy array

#to fiz numpy formt warning (b/c we didnt use iloc)
inputs = inputs.reshape(-1,1)

# scale inputs
inputs = sc.transform(inputs)

# Collecting the needed values
X_test = []
#20 financial days in January
for i in range(num_timesteps, num_timesteps + 20):
    X_test.append(inputs[i-num_timesteps:i, 0])
X_test = np.asarray(X_test)

# put inputs into 3d shape to account for times when you have more than one feature
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict future values
predicted_stock_price = regressor.predict(X_test)

#inverse scale to go from scaled predictions to the actual predicted prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting the results

# Creating the plots
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


