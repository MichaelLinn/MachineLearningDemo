# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:44:24 2016

@author: michael
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model


def get_data():
    data = pd.read_csv('E:/Spyder/LinearRegression/data/data.csv')
    X_parameter = []
    Y_parameter = []
    for x_data,y_data in zip(data['x'],data['y']):
        X_parameter.append([float(x_data)])
        Y_parameter.append(float(y_data))
    return X_parameter,Y_parameter
    
  
def linear_model_main(X_parameter,Y_parameter,predict_value):
    
    #Create linear regression object 
    regr =  linear_model.LinearRegression()
    regr.fit(X_parameter,Y_parameter)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions
    
def linear_model_manual(prediction_value):
    data = pd.read_csv('E://Spyder/LinearRegression/data/data.csv')
    X_tem = []
    Y_tem = []
    for X_data ,Y_data in zip(data['x'],data['y']):
        X_tem.append(int(X_data))
        Y_tem.append(float(Y_data))
    X_parameters = np.array(X_tem)
    Y_parameters = np.array(Y_tem)
    xy = X_parameters*Y_parameters
    xy_avg = xy.mean()
    x_avg = X_parameters.mean()
    y_avg = Y_parameters.mean()
    x_square = X_parameters*X_parameters
    x_square_avg = x_square.mean()
    predictions = {}
    #Method of least squares
    predictions['coefficient'] = (xy_avg - x_avg*y_avg) / (x_square_avg - x_avg*x_avg)
    predictions['intercept'] = y_avg - predictions['coefficient']*x_avg
    #prediction_result
    predictions['predictions_result'] = predictions['intercept'] + predictions['coefficient']*prediction_value    
    return predictions
    
    
def linear_model_multivariate():
    #coefficient = (X_trans*X)^-1 * X_trans * y 

    data = pd.read_csv('E://Spyder/LinearRegression/data/data.csv')
    X_tem = []
    Y_tem = []
    linearModel={}
    for X_data ,Y_data in zip(data['x'],data['y']):
        X_tem.append(int(X_data))
        Y_tem.append(float(Y_data))
    X_parameters = np.ones((len(X_tem),2))
    
    for i in range(len(X_tem)):
        X_parameters[i][0] = X_tem[i]

    Y_parameters = np.array(Y_tem)
    # Formula  
    # coefficient = inv(X.T*X) * X.T * y    
    coefficient = np.dot(np.dot(np.linalg.inv(np.dot(X_parameters.T,X_parameters)),X_parameters.T),Y_parameters)
     
    avg_X = X_parameters.mean(axis = 0)   
    intercept = Y_parameters.mean() + coefficient * avg_X[1]
    linearModel['coefficient'] = coefficient
    linearModel['intercept'] = intercept
    return linearModel
    
    
    
def get_loss():
    #Calculate the loss the linear_model
    data = pd.read_csv('E://Spyder/LinearRegression/data/data.csv')
    X_tem = []
    Y_tem = []
    
    for X_data ,Y_data in zip(data['x'],data['y']):
        X_tem.append([int(X_data)])
        Y_tem.append(float(Y_data))

    x_data = np.array(X_tem)
    y_data = np.array(Y_tem)
    
    regr = linear_model.LinearRegression() 
    regr.fit(x_data,y_data)
    loss = np.sum((y_data - regr.predict(x_data)) ** 2)
    return loss
        

    
#Function to show the result of linear fit model    
def show_linear_line(X_parameter,Y_parameter):
    #Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameter,Y_parameter)
    
    plt.scatter(X_parameter,Y_parameter,color='blue')
    plt.plot(X_parameter,regr.predict(X_parameter),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def main():
    print("the test is running");
    
if __name__ == '__main__':
    main()

    
    
    