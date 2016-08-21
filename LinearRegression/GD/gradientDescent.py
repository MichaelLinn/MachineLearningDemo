# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:27:11 2016

@author: michael
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class gradientDescent:
    numOfInterations = 100000
    step = 0.000005
    X_parameters = []
    Y_parameters = []
    length = 0   
    #coefficient matrix
    theta = np.ones(2)
    theta[0] = 1000
    theta[1] = 20
    #Loss
    loss = []
    
    def __init__(self,step,numOfInterations):
        data  = pd.read_csv('E:/Spyder/LinearRegression/data/data.csv')
        for x_data ,y_data in zip(data['x'] , data['y']):
            self.X_parameters.append(int(x_data))
            self.Y_parameters.append(float(y_data))
        self.length = len(self.X_parameters)
        self.numOfInterations = numOfInterations
        self.step = step
        
    def genData(self):
        # 求一元函数的 linear regression  y = a + b*x
        # 输入为两列 ，其中一列给 intercept*1
        self.genx = np.ones(shape = (self.length,2))
        self.geny = np.zeros(shape = self.length)
        for i in range(self.length):
            self.genx[i][1] = self.X_parameters[i]
            self.geny[i] = self.Y_parameters[i]
        
    def gradientDescent(self):
        # thetaJ = thetaJ + sum(y_i - theta_i*x_i)*xj  (GradientDescent)
        # transpose X matrix
        xTrans = self.genx.transpose()
        for i in range(self.numOfInterations):
            # h_theta(x) = theta * X_transpose
            hypothesis_theta = np.dot(self.theta,xTrans)
            # print hypothesis_theta
            loss_tem = self.geny - hypothesis_theta
            # square loss function
            self.loss.append(np.sum(loss_tem ** 2))
            # gradient = matrix_X * matrix(y_i - h_theta(x))
            #         = sum((y_i - h_theta(x_i)) * x_j)
            gradient = np.dot(xTrans , loss_tem)
            self.theta = self.theta + self.step * gradient       
            
    def gradientDescent2(self):
        xTrans = self.genx.transpose() #
        i = 0
        while i < self.numOfInterations:
            hypothesis = np.dot(self.genx,self.theta)
            loss = hypothesis - self.geny
            #record the cost
            self.cost.append(np.sum(loss ** 2))
            #calculate the gradient
            gradient = np.dot(xTrans,loss)
            #updata, gradientDescent
            self.theta = self.theta - self.step * gradient
            i = i + 1
            
    def show(self):        
        plt.scatter(self.X_parameters,self.Y_parameters,color = 'blue')
        y_result = np.add(self.theta[0] , np.dot(self.theta[1],self.X_parameters))
        plt.plot(self.X_parameters,y_result,color='red',linewidth=4)
        plt.xticks()
        plt.yticks()
        plt.show()


c = gradientDescent(0.0000005,5000000)
c.genData()
c.gradientDescent()
c.show()
for i in range(10):
    print(c.loss[-(i+1)])
print(c.theta)            
                
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
