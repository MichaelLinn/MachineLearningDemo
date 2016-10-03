# -*- coding: utf-8 -*-
# @Time     : 2016/10/2 22:52
# @Author   : Jason

import numpy as np
class randomForset():

    def __init__(self, train_data, n_estimators=10):
        self.data = train_data
        self.n_estimators = n_estimators
        self.decision_trees = []
        # the label means the fearure
        self.labels = np.unique(self.data[:, :-1])
        # gain the all the class from the data set
        # np.unique(self.data[:,-1])

    def cal_entropy(self,labels):
        labelSet = np.unique(labels)
        entropy = 0
        sum = len(labels)
        classLabels = {}
        for label in labels:
            classLabels[label] = classLabels.get(label,0) + 1
        for classLabel in labelSet:
            p = classLabels[classLabel]*1.0/sum
            entropy += -p*np.log2(p)
        return entropy

    def split_data(self,dataSet,bestFeature,value):
        split_data = []
        for vector in dataSet:
            if vector[bestFeature] == value:
                split_data.append(vector[:bestFeature].toList() + vector[bestFeature+1:].toList())
        return np.array(split_data)

    def choose_bestFeature(self,dataSet):
        classLabels = dataSet[:,-1]
        featureSet = dataSet[:,:-1]
        baseEntropy = self.cal_entropy(classLabels)
        infoGain = -1
        bestFeature = -1
        uniqueClassLabels = np.unique(classLabels)
        len_dataSet,n_col = dataSet.shape()
        featureNum = len(featureSet[0])
        for feature in range(featureNum):
            empirical_entropy = -1
            featureValue = np.unique(dataSet[:,feature])
            for value in featureValue:
                split_data = self.split_data(dataSet,feature,value)
                empirical_entropy = len(split_data)*1.0/len_dataSet*self.cal_entropy(split_data,featureValue,value)
                temInfoGain = baseEntropy - empirical_entropy
            if temInfoGain > infoGain:
                infoGain = temInfoGain
                bestFeature = feature
        return bestFeature,infoGain

    def build_ID3DecisionTree(self,dataSet):
        # conditions for termination of the recurcive function
        # condition one : there is only one label class in the data set
        decisionTree = {}
        classLabel = dataSet[:,-1]
        if len(np.unique(classLabel)) == 1:
            return classLabel[0]
        # condition two : there is no feature to split
        if len(dataSet[0]) == 1:
            sum_classlabel = {}
            classLabels = dataSet[:,-1]
            for label_class in classLabels:
                sum_classlabel[label_class] = sum_classlabel.get(label_class,0) + 1
            l = sorted(sum_classlabel.items(), key=lambda x:x[1], reverse=True)
            return l[0][0]
        bestFeature = self.choose_bestFeature(dataSet)
        value_bestFeature = np.unique(dataSet[:,bestFeature])
























