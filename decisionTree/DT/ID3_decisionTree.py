# -*- coding: utf-8 -*-

"""
Created on Thu Jul 28 12:36:46 2016

@author: michael
"""

from math import log

class ID3_decisionTree:

    dataSet = []

    def __init__(self,dataSet):
        self.dataSet = dataSet


    def calcShannonEnt(self):
        numEntries = len(self.dataSet)
        labelCounts = {}
        for featVec in self.dataSet:
            currentLabel = featVec[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]/numEntries)
            shannonEnt += -(prob*log(prob,2))
        return shannonEnt

#在数据集D中去掉特征值A，方便计算基于属性A_i划分后的数据集D_n的熵
    def splitDataSet(self,dataSet,axis,value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reduceFeatVec)
        return retDataSet


def main():
    print("the test is running!")
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    dt = ID3_decisionTree(dataSet)
    print(dt.calcShannonEnt())

if __name__ == '__main__':
    main()









