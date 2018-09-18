#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 14:27:31 2018

@author: neilklenk
"""
import numpy as np
import pandas as pd

def wrangle_data(df, num_timesteps, lookback_window, skip_scale='No'):
    # Import data
    dataset_train = df['adjClose'][:-lookback_window]
    training_data = np.reshape(dataset_train, (-1, 1))
    #dataset_train.iloc[:, 1:2].values    # .values turns series obj into an array
    
    # Feature scaling - maxmin average vs standard scaling w/ distribution
    from sklearn.preprocessing import MinMaxScaler
    # scale all values between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = training_data
    if skip_scale == 'No':
        training_set_scaled = sc.fit_transform(training_data)
    
    # creating a data structure with 60 timesteps and one output
    #num_timesteps = 60
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
    
    return (X_train, y_train, sc, dataset_train)
    

def wrangle_data_1d(df, num_timesteps, lookback_window):
    # Import data
    dataset_train = df['adjClose'][:-lookback_window]
    training_data = np.reshape(dataset_train, (-1, 1))
    #dataset_train.iloc[:, 1:2].values    # .values turns series obj into an array
    
    # Feature scaling - maxmin average vs standard scaling w/ distribution
    from sklearn.preprocessing import MinMaxScaler
    # scale all values between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_data)
    
    # creating a data structure with 60 timesteps and one output
    #num_timesteps = 60
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
    
    return (X_train, y_train, sc, dataset_train)

def moving_average(df, column='adjClose', days_in_average=20):
    df_train = np.zeros(shape=(len(df[column]) - days_in_average, 1))
    index_ma=[]
    for i in range(len(df[column]) - days_in_average):
        df_train[i] = np.average(df[column][i:days_in_average+i])
    for i in range(len(df_train)):
        index_ma.append(i)
        
    return (pd.DataFrame(df_train, columns=[column]))

#def 
#
#def 200_day_moving_average():
#    return None
#
#
#def wrangle_data_3d(df, num_timesteps):
#    return None
#    

    