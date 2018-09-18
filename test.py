
#7ba8b224ddf9d253c34b264f22be8#!/usr/bin/env python3
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

# Keras packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import model_from_json

# Bokeh plotting 
from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file
    
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
    json_out_file = "regressor"+stock+".json"
    with open(json_out_file, "w") as json_file:
        json_file.write(regressor_json)
    # serialize weights to HDF5
    regressor_out_weights = "regressor"+stock+".h5"
    regressor.save_weights(regressor_out_weights)
    print("Saved model to disk")
     
    # later...
     
    # load json and create model
    json_in_file = "regressor"+stock+".json"
    json_file = open(json_in_file, 'r')
    loaded_regressor_json = json_file.read()
    json_file.close()
    loaded_regressor = model_from_json(loaded_regressor_json)
    # load weights into new model
    regressor_in_weights = "regressor"+stock+".h5"
    regressor.load_weights(regressor_in_weights)
    print("Loaded model from disk")
    
    ## Making Predictions and visualizing the results
    
    # Getting the true stock prices for 2017
    dataset_test= df['adjClose'][-20:]
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
    dataset_newindex = dataset_test.reset_index()
    date_lables = pd.DatetimeIndex(dataset_newindex['date']).strftime("%y/%m/%d")
    dates = pd.DatetimeIndex(dataset_newindex['date'])

    
    stock_plot = figure(x_axis_type='datetime', title=stock, plot_width=400, plot_height=400)
    stock_plot.xaxis.axis_label = 'Date'
    stock_plot.yaxis.axis_label = 'Value in Dollars'
    stock_plot.line(dates, dataset_test.values, color='blue', line_width=2, legend='Actual Values')
    stock_plot.circle(dates, dataset_test.values, color='blue')
    stock_plot.line(dates, list(predicted_stock_price.flatten()), color='red', line_width=2, legend='Predicted Values' )
    stock_plot.circle(dates, list(predicted_stock_price.flatten()), color='red')
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
grid = gridplot([[p1, p2], [p3, p4], [p5, p6]])
try:
    show(grid)
except AttributeError:
    show(grid)



# Import smtplib for the actual sending function
import smtplib


# Here are the email package modules we'll need
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sender = 'neil.klenk@gmail.com'
receivers = ['neil.klenk@gmail.com']#, 'johnk9000@gmail.com']

# Create the container (outer) email message.
msg = MIMEMultipart()
msg['Subject'] = 'Todays Stock Preds'
# me == the sender's email address
# family = the list of all recipients' email addresses
msg['From'] = 'neil.klenk@gmail.com'
msg['To'] = 'neil.klenk@gmail.com' #COMMASPACE.join(family)
msg.preamble = 'Our family reunion'


filename = "stock_plots.html"
f = open(filename)
attachment = MIMEText(f.read(), _subtype='html')
attachment.add_header('Content-Disposition', 'attachment', filename=filename)
msg.attach(attachment)

gmail_sender = 'neil.klenk@gmail.com'
gmail_passwd = 'earMark23!!'

# Send the email via our own SMTP server.
server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.login(gmail_sender, gmail_passwd)
print ("Successfully sent email")

server.sendmail(sender, receivers, msg.as_string()) #
server.quit()

