# -*- coding:utf-8 -*- 
"""
Created on '2016/9/3' '21:01'

@author: 'michael"  
"""
import numpy as np

class k_Means:

    def __init__(self):
        a = 3


    def loadDataSet(self,filename):
        dataMat = []
        fr = open(filename)
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = map(float,curLine)
            dataMat.append(fltLine)
        return dataMat

    def distEclud(self, vecA , vecB):
        return  np.sqrt(sum(np.power(vecA - vecB , 2)))

    def randCenter(self , dataSet , k):
        n = np.shape(dataSet)[1]   # the number of attributes in dataSet
        centroids = np.mat(np.zeros((k,n)))     # the average points of the k-clusters
        for j in range(n):                      # initialize the mean points of the k-clusters
            minJ = min(dataSet[:,j])
            rangJ = float(max(dataSet[:,j] - minJ))
            centroids[:,j] = minJ + rangJ * np.random.rand(k,1)
        return centroids


    def kMeans(self, dataSet,k, distMeans = distEclud , createCent = randCenter):
        m = np.shape(dataSet)[0]              # the number of the vectors in dataSet
        clusterAssment =  np.mat(np.zeros((m,2)))
        centroids =  createCent(dataSet, k)
        clusterChanged  = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                for j in range(k):
                    distJI = distMeans(centroids[j,:] , dataSet[i,:])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i,:] != minIndex:
                    clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2
            print(centroids)
            for cent in range(k):
                ptsInClust = dataSet[np.nozero(clusterAssment[:,0].A == cent)[0]]
                centroids[cent,:] = np.mean(ptsInClust, axis = 0)
        return centroids, clusterAssment































