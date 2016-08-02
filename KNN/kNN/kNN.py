# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:36:46 2016

@author: michael
"""

import numpy as np

class kNN:    
    
        
    
    def __init__(self):
        self.filename = 'E:\Spyder\KNN\data\datingTestSet.txt'
        
        
        
        
    
    
    def kNN_classfiy(self,dataset,labels,k):
        return 0
        
        
        
    
    def file2matrix(self,filename):
        fr = open(filename)
        arrayOfLines = fr.readlines()
        returnMat = np.zeros((len(arrayOfLines),3))
        #The class of data
        classLabelVector = []
        index = 0
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[:3]
            classLabelVector.append(listFromLine[-1])
            index += 1
        return returnMat,classLabelVector
            
    def autoNorm(dataset):
            
    
            
c = kNN()
print(c.file2matrix(c.filename))
            
            
        
        
        
        
    
    