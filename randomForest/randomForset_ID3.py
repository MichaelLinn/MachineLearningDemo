# -*- coding: utf-8 -*-
# @Time     : 2016/10/2 22:52
# @Author   : Jason

import numpy as np
import random
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

    def build_ID3DecisionTree(self,dataSet,featureNameList):
        # featureName is a set including all the name of the feature
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
        decisionTree = {bestFeature:{}}
        featureNameList.remove(bestFeature)
        for single_value in value_bestFeature:
            decisionTree[bestFeature][single_value] = self.build_ID3DecisionTree(self.split_data(bestFeature,single_value),featureNameList)
        return decisionTree

    def fit(self):
        print(self.data.shape)
        num_data,num_feature = self.data.shape
        for i in num_data:
            # sample data
            samples = np.random.randint(0,num_data,(num_data+1)/2)
            samples_data = self.data[samples.tolist()]
            # sample feature select log2(num_feature) feature from the feature set for
            # every tree
            features = random.sample(range(num_feature-1),int(np.log2(num_feature-1)))
            feature_list = features.tolist()
            feature_list.append(-1)
            samples_feature_data = samples_data[:,feature_list]
            # build every single tree
            decision_tree = self.build_ID3DecisionTree(samples_feature_data,feature_list)
            # all single trees converge into a random forest
            self.decision_trees.append(decision_tree)

    def classify(self,decision_tree,test_x):   # One tree classifier
        first_feature = decision_tree.keys()[0]
        secondDict = decision_tree[first_feature]
        feature = test_x[first_feature]
        value = secondDict.get(feature,"false")
        # if we can no find the key,then we will return a label randomly
        if feature == "false":
            return "Cannnot classify"
            values = secondDict.values()
            r = np.random.randint(0,len(values))
            return values[r]
        if isinstance(value,dict):  # judge if the value is leaf or tree node
            classLabel = self.classify(value,test_x)
        else:
            classLabel = value
        return classLabel

    def predict(self,test_data):
        predict_labels=[]
        for test_x in test_data:
            dic={}
            for decision_tree in self.decision_trees:
                predict_label = self.classify(decision_tree,test_x)
                if predict_label == 'Cannot classify':
                    continue
                dic[predict_label] = dic.get(predict_label,0) + 1
            l = sorted(dic.items(),key = lambda x:x[1],reverse = True)
            # if test_data has the unknown feature
            if len(l) == 0:
                r = np.random.randint(0,self.labels.shape[0])
                label = self.labels[r]
                predict_label.append(label)
            else:
                predict_labels.append(l[0][0])
        return np.array(predict_labels)

    def accuracy(self,test_data):
        predict_labels = self.predict(test_data[:,:-1])
        length = test_data.shape[0]
        num = 0
        for i in range(length):
            if predict_labels[i] == test_data[i,-1]:
                num += 1
        print("Test data accuracy is %f"%(num*1.0/length))






































