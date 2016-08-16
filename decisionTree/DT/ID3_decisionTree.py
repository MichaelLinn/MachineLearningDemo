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

#基于特征值A，把D分成D_1,D_2,...,D_i,...,D_n, 此处计算 H(D_i|A = a_i)
# empirical entropy = sum_i( p_i * H(D|A = a_i) )
#                   = sum_i ( |D_i|/|D| * sum_k( - (|D_ik| / |D_i|) log (|D_ik| / |D_i|) ))
    def calcShannonEnt(self,dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel,0) + 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]/numEntries)
            shannonEnt += -(prob*log(prob,2))
        return shannonEnt

#在数据集D中去掉特征值A_i，方便计算基于属性A_i划分后的数据集D_n的熵
    def splitDataSet(self,dataSet,axis,value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reduceFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self,dataSet):
#       特征值数量
        numFeatures = len(dataSet[0] - 1)
        baseEntropy = self.calcShannonEnt(self.dataSet)
        baseInfoGain = 0.0
        beastFeature = -1
        for i in numFeatures:
            featList = [example[i] for example in dataSet]
        # 用集合的形式存储属性A的所有取值，重复的值只保存一次
            uniqueVals = set(featList)










def main():
    print("the test is running!")
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    dt = ID3_decisionTree(dataSet)
    print(dt.calcShannonEnt())

if __name__ == '__main__':
    main()









