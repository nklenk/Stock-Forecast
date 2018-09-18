#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Mon Feb 12 18:23:47 2018

@author: neilklenk
"""
# Future
# predict a few days out by setting the pldest values to the mean of the remainder
# Bokeh plots
# Fix time scale on bottom
# Send plots to a website

#import packages
import numpy as np
import pandas_datareader.data as pdr

# Keras packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import model_from_json
from sklearn.externals import joblib

# dictionary for plots 
plot_dict = {}
stocks = ['HD']
#stocks = ['MU', 'TMO', 'CELG', 'NVDA', 'AQ', 'HD']


for c, stock in enumerate(stocks):   
    df = pdr.get_data_tiingo(stock, api_key='7ba8b224ddf9d253c34b264f22be8c9967206b3c')
    
    # Import data
    dataset_train = df['adjClose'][:-20]
    training_data = np.reshape(dataset_train, (-1, 1))
    #dataset_train.iloc[:, 1:2].values    # .values turns series obj into an array
    
    # Feature scaling - maxmin average vs standard scaling w/ distribution
    from sklearn.preprocessing import MinMaxScaler
    # scale all values between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_data)
    
    #Save scaler
    scaler_filename = "scaler_"+stock+".save"
    joblib.dump(sc, scaler_filename) 
    
    # creating a data structure with 60 timesteps and one output
    #num_timesteps = 60
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
    regressor.fit(X_train, y_train, epochs = 1, batch_size = 32)
    
    # Saving the model as a JSON file
    regressor_json = regressor.to_json()
    json_out_file = "regressor_"+stock+".json"
    with open(json_out_file, "w") as json_file:
        json_file.write(regressor_json)
    # serialize weights to HDF5
    regressor_out_weights = "regressor_"+stock+".h5"
    regressor.save_weights(regressor_out_weights)
    print("Saved model to disk")
     