#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:50:47 2018

@author: neilklenk
"""
# Future
# currently does not show final predicted day
# Bokeh plots
# Fix time scale on bottom
# Send plots to a website
#check for adequate numer of points to plot

#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json

from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file

from Email import *
from Preprocessing import *
from BuildModel import *
from LookAhead import *
from model_reuse import *


# currently does not show final predicted day
    
# dictionary for plots 
plot_dict = {}
stocks = ['HD']
#stocks = ['MU', 'TMO', 'CELG', 'NVDA', 'TSLA', 'HD']
num_timesteps = 60
lookback_window = 90
forecast_window = 5 #days
 
for c, stock in enumerate(stocks):
    print('Building Stock: {}'.format(stock))
    df = pdr.get_data_tiingo(stock, api_key='7ba8b224ddf9d253c34b264f22be8c9967206b3c')
    
    X_train, y_train, sc, dataset_train = wrangle_data(df, num_timesteps, lookback_window)
    
    X_train_20MA_prep = moving_average(df)
    X_train_200MA_prep = moving_average(df, days_in_average=200)
    
    X_train_20MA, y_train_20MA, sc_20MA, dataset_train_20MA = wrangle_data(X_train_20MA_prep, num_timesteps, lookback_window, skip_scale='Yes')
    X_train_200MA, y_train_200MA, sc_200MA, dataset_train_200MA = wrangle_data(X_train_200MA_prep, num_timesteps, lookback_window, skip_scale='Yes')
    
    MA20_delta = len(X_train_20MA) - len(X_train_200MA)
    MA0_delta = len(X_train) - len(X_train_200MA)
    
    X_train_20MA = X_train_20MA[MA20_delta:]
    X_train = X_train[MA0_delta:]
    
    #sc_MA = MinMaxScaler(feature_range=(0, 1))
    sc_MA = MinMaxScaler(feature_range=(-1, 1))
    
    Moving_average_delta = X_train_20MA - X_train_200MA
    Moving_average_delta = sc_MA.fit_transform(np.reshape(Moving_average_delta, (Moving_average_delta.shape[0], Moving_average_delta.shape[1])))
    Moving_average_delta = np.reshape(Moving_average_delta, (Moving_average_delta.shape[0], Moving_average_delta.shape[1], 1))
    
    # Remove n_timeseteps most recent data for testing later
    X_train_final = np.append(X_train[:-lookback_window], Moving_average_delta[:-lookback_window], axis=2)
    y_train_final = y_train[MA0_delta:-lookback_window]
    
    X_test_final= np.append(X_train[-lookback_window:], Moving_average_delta[-lookback_window:], axis=2)
    y_test_final = y_train[-lookback_window:]

    
    regressor = build_regressor(X_train_final, y_train_final, X_train_final.shape[2])
    # serialize model to JSON
    
    # Save model 
    save_model(regressor, stock, model_type = 'regressor')
    # Load model
    #load_model()
    
    # Getting the true stock prices for 2017
    dataset_test= df['adjClose'][-lookback_window:]
    real_stock_price = np.reshape(dataset_test, (-1, 1))
    
    # Predict next days values 
    predicted_stock_price = regressor.predict(X_test_final)
    #Predictions are for the following day
    predicted_stock_price_plot = np.append([predicted_stock_price[0]], predicted_stock_price, axis =0)
    
    # Look ahead prediction
    #look_ahead_price = look_ahead(regressor, X_test)
    window = forecast_window #days
    
    #** Changing X_test to X_test_final
    five_point_pred = []
    test = []
    num_breaks = int(len(X_test_final)/window)
    for i in range(num_breaks):
        current_vec = X_test_final[i*window:i*window+1]
        for j in range(window):
            test.append(current_vec)
            #print(current_vec.shape[])
            new_pred = np.reshape(regressor.predict(current_vec), (1,1,1))
            new_MA = np.reshape(np.mean(current_vec[:,:,1]), (1,1,1))
            new_values = np.concatenate((new_pred, new_MA), axis=2)
            vec_append = current_vec[:,1:current_vec.shape[1]]
            current_vec = np.append(vec_append, new_values, axis = 1)
            #current_vec = np.reshape(current_vec, (1,current_vec.shape[0], 1))
            five_point_pred.append(new_pred.flatten())
    
    
    #inverse scale to go from scaled predictions to the actual predicted prices
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    five_point_pred = sc.inverse_transform(np.asarray(five_point_pred))
    # Plotting the result
    
    dataset_newindex = dataset_test.reset_index()
    date_lables = pd.DatetimeIndex(dataset_newindex['date']).strftime("%y/%m/%d")
    dates = pd.DatetimeIndex(dataset_newindex['date'])
    
    y_test_final = sc.inverse_transform(y_test_final.reshape(-1,1))
    
    stock_plot = figure(x_axis_type='datetime', title=stock, plot_width=400, plot_height=400)
    stock_plot.xaxis.axis_label = 'Date'
    stock_plot.yaxis.axis_label = 'Value in Dollars'
    stock_plot.line(dates, y_test_final.flatten(), color='blue', line_width=2, legend='Actual Values')
    stock_plot.circle(dates, y_test_final.flatten(), color='blue')
    stock_plot.line(dates, list(predicted_stock_price.flatten()), color='red', line_width=2, legend='Point-Predicted Values' )
    stock_plot.circle(dates, list(predicted_stock_price.flatten()), color='red')
    stock_plot.line(dates, five_point_pred.flatten(), color='green', line_width=2, legend='Predicted Values to {} Days'.format(forecast_window) )
    stock_plot.circle(dates, five_point_pred.flatten(), color='green')
    #stock_plot.legend=False
    #stock_plot.legend.location = "top_left"
    #stock_plot.legend.click_policy="hide"

    plot_dict[stock] = stock_plot

p1 = plot_dict[stocks[0]]
p2 = plot_dict[stocks[1]]
p3 = plot_dict[stocks[2]]
p4 = plot_dict[stocks[3]]
p5 = plot_dict[stocks[4]]
p6 = plot_dict[stocks[5]]
    
output_file("stock_plots.html", title="20 days of values")    
#grid = gridplot([[p1]])
grid = gridplot([[p1, p2], [p3, p4], [p5, p6]])

try:
    show(grid)
except AttributeError:
    show(grid)

email_results()
