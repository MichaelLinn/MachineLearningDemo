# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:09:18 2016

@author: michael
"""

import numpy as np
import pandas as pd
import csv


class handWritingClassifier:
    
    def __init__(self):
        trainFileName = 'E:/Spyder/KNN/kaggleHandWritingData/train.csv'
        testFileName = 'E:/Spyder/KNN\kaggleHandWritingData/test.csv'
        self.trainDataFrame = pd.read_csv(trainFileName)
        self.testDataFrame = pd.read_csv(testFileName)
        
    def file2matrix(self):
         self.trainDataLabels = self.trainDataFrame[[0]].values.ravel()
         self.trainDataSet = self.trainDataFrame.iloc[:,1:].values
         self.testDataSet = self.testDataFrame.values
         
    def kNN_classify(self,testData,labels,trainData,k):
      
        length = len(labels)
        testDataMat = np.tile(testData,(length,1))
        diffMat = testDataMat - trainData
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis = 1)
        distances = sqDistances ** 0.5
        classCount = {}
        sortedDistancesIndex = np.argsort(distances)
        for i in range(k):
            voteLable = labels[sortedDistancesIndex[i]]
            classCount[voteLable] = classCount.get(voteLable,0) + 1
        classCount = sorted(classCount.items(), key = lambda classCount:classCount[1] , reverse = True)
        return classCount[0][0]
        
    def handWritingClassfy(self):
        
#        testHandWritingLables = []
        with open('../kaggleHandWritingData/submission.csv','w',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(('ImageId','Label'))
            length = len(self.testDataSet)
            for i in range(length):
                testResult = self.kNN_classify(self.testDataSet[i,:],self.trainDataLabels[:],self.trainDataSet[:,:],3)
#               testHandWritingLables.append(testResult)
                print("No.%d:the classifier result come back with %s" % (i+1 , testResult))
                csvwriter.writerow((i+1,testResult))
#            imageId = np.arange(i+2)
#            result['ImageId'] = imageId[1:i+2]
#            result['Label'] = testHandWritingLables
#            hwClassifierTestingResult = pd.DataFrame(result)
#            hwClassifierTestingResult.to_csv('../kaggleHandWritingData/submission.csv' , index = False)
            
        
        
        
    def testHandWritingClassifier(self):
        testRatio = 0.01
        lenSum = len(self.trainDataLabels)
        testSum = int(lenSum * testRatio)
        testLabels= []
        errorCount = 0.0
        for i in range(testSum):
            testResult = self.kNN_classify(self.trainDataSet[i,:], \
                                           self.trainDataLabels[testSum:],self.trainDataSet[testSum:,:],5)
            testLabels.append(testResult)
            print("No.%d : the classifier result come back with %s , the real result is %s " %(i,testResult, self.trainDataLabels[i]))
            if(testResult != self.trainDataLabels[i]): errorCount += 1.0
        errorRate = errorCount/testSum
        print("the error count of handwriting classifier is : %f " %errorCount)
        print("the error rate of handwriting classifier is : %f " %errorRate)
        
def main():
    print("the test is running")
    hw = handWritingClassifier()
    hw.file2matrix()
    hw.handWritingClassfy()
    
if __name__ == '__main__':
    main()
    
        
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
         
         
        
        