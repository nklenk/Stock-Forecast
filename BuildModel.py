#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 14:31:02 2018

@author: neilklenk
"""

# Keras packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def build_regressor(X_train, y_train, n_features = 1):   
    # Initialize model
    # Regressor because we are looking for a number
    regressor = Sequential()
    
    # Adding first LSTM layer and some dropout regularization
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], n_features)))
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
    
    return (regressor)