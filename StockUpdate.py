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
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr

from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file

from Email import *
from Preprocessing import *
from BuildModel import *
from LookAhead import *

#predict future movements

# currently does not show final predicted day
    
# dictionary for plots 
plot_dict = {}
#stocks = ['HD']
stocks = ['MU', 'TMO', 'CELG', 'NVDA', 'AQ', 'HD']
num_timesteps = 60
lookback_window = 80
 
for c, stock in enumerate(stocks):   
    df = pdr.get_data_tiingo(stock, api_key='7ba8b224ddf9d253c34b264f22be8c9967206b3c')
    
    X_train, y_train, sc, dataset_train = wrangle_data(df, num_timesteps)
    
    regressor = build_regressor(X_train, y_train)
    
    ## Making Predictions and visualizing the results
    
    # Getting the true stock prices for 2017
    dataset_test= df['adjClose'][-lookback_window:]
    real_stock_price = np.reshape(dataset_test, (-1, 1))
    
    # Getting the predicted stock price of 2017
    # Need to conctentat training set onto test set, but must scale test set in the same way we scaled the train set 
    # to make a prediction on instance t, you need [t-60:t-1] previous instances
    
    # concatenate origional train and test sets
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    
    
    # take the last day, and previous 60 days as out test 
    inputs = dataset_total[len(dataset_total) - (len(dataset_test) + num_timesteps):].values #numpy array
    
    #to fiz numpy formt warning (b/c we didnt use iloc)
    inputs = inputs.reshape(-1,1)
    
    #inputs = np.char.strip(inputs, ',')
    # scale inputs
    
    inputs = sc.transform(inputs)
    
    # Collecting the needed values
    X_test = []
    for i in range(num_timesteps, num_timesteps + lookback_window):
        X_test.append(inputs[i-num_timesteps:i, 0])
        if i < 5:
            print(inputs[i-num_timesteps:i, 0])
    X_test = np.asarray(X_test)
    
    # put inputs into 3d shape to account for times when you have more than one feature
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predict next days values 
    predicted_stock_price = regressor.predict(X_test)
    #Predictions are for the following day
    predicted_stock_price_plot = np.append([predicted_stock_price[0]], predicted_stock_price, axis =0)
    
    # Look ahead prediction
    #look_ahead_price = look_ahead(regressor, X_test)
    window =5
    
    five_point_pred = []
    test = []
    num_breaks = int(len(X_test)/window)
    for i in range(num_breaks):
        current_vec = X_test[i*window:i*window+1]
        for j in range(window):
            test.append(current_vec)
            #print(current_vec.shape[])
            new_pred = regressor.predict(current_vec)
            vec_append = current_vec[:,1:current_vec.shape[1]]
            current_vec = np.append(vec_append, np.reshape(new_pred, (1,1,1)))
            current_vec = np.reshape(current_vec, (1,current_vec.shape[0], 1))
            five_point_pred.append(new_pred.flatten())
    
    
    #inverse scale to go from scaled predictions to the actual predicted prices
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    five_point_pred = sc.inverse_transform(np.asarray(five_point_pred))
    # Plotting the result
    
    #Creating the plots
#    plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
#    plt.plot(predicted_stock_price_plot, color = 'blue', label = 'Predicted Stock Price')
#    plt.title(stock+'Price Prediction')
#    plt.xlabel('Time')
#    plt.ylabel('Price')
#    plt.legend()
#    plt.show()
    
    

    dataset_newindex = dataset_test.reset_index()
    date_lables = pd.DatetimeIndex(dataset_newindex['date']).strftime("%y/%m/%d")
    dates = pd.DatetimeIndex(dataset_newindex['date'])

    
    stock_plot = figure(x_axis_type='datetime', title=stock, plot_width=400, plot_height=400)
    stock_plot.xaxis.axis_label = 'Date'
    stock_plot.yaxis.axis_label = 'Value in Dollars'
    stock_plot.line(dates, dataset_test.values, color='blue', line_width=2, legend='Actual Values')
    stock_plot.circle(dates, dataset_test.values, color='blue')
    stock_plot.line(dates, list(predicted_stock_price.flatten()), color='red', line_width=2, legend='Point-Predicted Values' )
    stock_plot.circle(dates, list(predicted_stock_price.flatten()), color='red')
    stock_plot.line(dates, five_point_pred.flatten(), color='green', line_width=2, legend='Predicted Values to 20 Points' )
    stock_plot.circle(dates, five_point_pred.flatten(), color='green')
    stock_plot.legend.location = "top_left"
    stock_plot.legend.click_policy="hide"

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

