# -*- coding:utf-8 -*- 
"""
Created on '2016/9/9' '10:32'

@author: 'michael"  
"""

import numpy as np

class adaBoost:

    def __init__(self):
        a = 1

    def stumpDecisionTree(self,dataMat,dimen, thresholdVal, thresholdIneq):
        retArray = np.ones((np.shape(dataMat)[0],1))   # nunmber of the vector in dataMatrix
        if thresholdVal == 'lt':                       # two kinds of the inequal type
            retArray[dataMat[:,dimen] <= thresholdVal] = -1.0
        else:
            retArray[dataMat[:,dimen] > thresholdVal] = -1.0
        return retArray

    def buildStump(self,dataArr,classLabels,D):
        dataMatrix = np.mat(dataArr)
        m,n = np.shape(dataMatrix)
        numSteps = 10.0
        bestStump = {}
        bestClassEst = np.mat(np.zeros((m,1)))
        minError = np.inf
        for i in range(n):
            rangeMin = dataMatrix[:,i].min()
            rangeMax = dataMatrix[:,i].max()
            stepSize = (rangeMax - rangeMin)/numSteps
            for j in range(-1,int(numSteps) + 1):
                for inequal in ['lt' , 'gt']:
                    thresholdVal = (rangeMin + float(j) * stepSize)
                    predictedVals = self.stumpDecisionTree(dataMatrix,i,thresholdVal,inequal)
                    errArr = np.mat(np.ones((m,1)))
                    errArr[ predictedVals == classLabels] = 0
                    weightedError = D.T * errArr

                    if weightedError < minError:
                        minError = weightedError
                        bestClassEst = predictedVals.copy()
                        bestStump['dimension'] = i
                        bestStump['inequal'] = inequal
                        bestStump['threshold'] = thresholdVal
        return bestStump,minError,bestClassEst

    def adaBoostTrainDecisionStump(self,dataArr,classLabels,numInt=40):
        weekclassArr = []
        m = np.shape(dataArr)[0]
        D = np.mat(np.ones((m,1))/m)
        aggressionClassEst = np.mat(np.zeros((m,1)))
        for i in range(numInt):
            bestStump,error,classEst = self.buildStump(dataArr,classLabels)


























