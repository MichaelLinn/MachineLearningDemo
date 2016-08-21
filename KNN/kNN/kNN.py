# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:36:46 2016

@author: michael
"""

import numpy as np
# 得到目录下所有文件名 get all the filename in the directory
from os import listdir


class kNN:
    def __init__(self):
        self.filename = '..\data\datingTestSet.txt'
        # 之前出现的问题就是 输入没有进归一化处理 导致不同输入的结果没有什么变化
        self.inX = np.array([1200, 1340, 0.9])

#   k-nearest-neighbor
    def kNN_classfiy(self, intX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        # np.tile() method makes the input vector into the same size of the dataSet
        # in order to do the subtraction between the every sub vector in intX and the all the training vectors
        #  at one time
        diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        # sum up all the subtractions of every subtraction to get Euler distances between the intX and all vectors
        # in dataSet
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        # 返回按照大小排序的数组的index
        sortedDistancesIndexs = distances.argsort()
        classCount = {}
        for i in range(k):
            voteILable = labels[sortedDistancesIndexs[i]]
            # count the class
            # usage : clasCount[voteILable] 存在则 classCount[voteILable] += 1
            #                               否则  classCount[voteILabel] = 0 + 1
            classCount[voteILable] = classCount.get(voteILable, 0) + 1

        sortedClassCount = sorted(classCount.items(), key=lambda classCount: classCount[1], reverse=True)
        return sortedClassCount[0][0]

    def file2matrix(self, filename):
        fr = open(filename)
        arrayOfLines = fr.readlines()
        returnMat = np.zeros((len(arrayOfLines), 3))
        # The class of data
        classLabelVector = []
        index = 0
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[:3]
            classLabelVector.append(listFromLine[-1])
            index += 1
        self.classLabelVector = classLabelVector
        return returnMat, classLabelVector

#   normalize all the attributes
    def autoNorm(self, dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = np.zeros(np.shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - np.tile(minVals, (m, 1))
        normDataSet = normDataSet / np.tile(ranges, (m, 1))
        self.normDataSet = normDataSet
        self.ranges = ranges
        self.minVals = minVals
        # print(normDataSet)
        return normDataSet, ranges, minVals

    def datingClassTest(self):
        hoRatio = 0.1
        m = self.normDataSet.shape[0]
        numTestVecs = int(m * hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = self.kNN_classfiy(self.normDataSet[i, :], self.normDataSet[numTestVecs:m, :], \
                                                 self.classLabelVector[numTestVecs:m], 3)
            print("the classifierResult came back with: %s, the real answer is : %s" \
                  % (classifierResult, self.classLabelVector[i]))
            if (classifierResult != self.classLabelVector[i]):
                errorCount += 1.0
        print("the total error rate is : %f" % (errorCount / float(numTestVecs)))
        print(errorCount)


    def img2vector(self, filename):
        returnVector = np.zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVector[0, 32 * i + j] = int(lineStr[j])
        return returnVector

    def handwritingClassTest(self):
        hwLabels = []
        trainingFileList = listdir('E:/Spyder/KNN/trainingDigits/')
        m = len(trainingFileList)
        trainingMat = np.zeros((m, 1024))
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i, :] = self.img2vector('E:/Spyder/kNN/trainingDigits/%s' % fileNameStr)
        testFileList = listdir('E:/Spyder/KNN/testDigits/')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = self.img2vector('E:/Spyder/KNN/testDigits/%s' % fileNameStr)
            classifierResult = self.kNN_classfiy(vectorUnderTest, trainingMat, hwLabels, 5)
            print("the classifier came back with : %d , the real answer is : %d" % (classifierResult, classNumStr))
            if (classifierResult != classNumStr): errorCount += 1.0
        print("the total number of error is : %d" % errorCount)
        print("the total error rate is :%f " % (errorCount / float(mTest)))


c = kNN()
dataSet, labels = c.file2matrix(c.filename)
normDataSet, ranges, minVals = c.autoNorm(dataSet)
normInput = (c.inX-minVals)/ranges
print(normInput)
print(c.kNN_classfiy(normInput , dataSet , labels,10))

# c.datingClassTest()
# c.handwritingClassTest()













