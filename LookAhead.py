#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:27:39 2018

@author: neilklenk
"""
import numpy as np

def look_ahead(regressor, X_test, window=20):
    pred = []
    num_breaks = int(len(X_test)/window)
    for i in range(num_breaks):
        current_vec = X_test[i*20]
        for j in range(window):
            print(current_vec.shape)
            current_vec = np.reshape(current_vec, (current_vec.shape[0], current_vec.shape[1], 1))
            new_pred = regressor.predict(current_vec[:])
            current_vec = np.append([new_pred], current_vec[0:len(current_vec)], axis = 0)
            pred.append(new_pred)
    
    return(pred)
        
    