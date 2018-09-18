#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 17:17:41 2018

@author: neilklenk
"""
def save_model(model, stock, model_type='regressor'):
    regressor_json = model.to_json()
    with open("{}_model_{}.json".format(model_type, stock), "w") as json_file:
        json_file.write(regressor_json)
    # serialize weights to HDF5
    model.save_weights("{}_weights_{}.h5".format(model_type, stock))
    print("Saved model for {} to disk".format(stock))