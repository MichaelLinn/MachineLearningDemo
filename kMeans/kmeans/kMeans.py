# -*- coding:utf-8 -*- 
"""
Created on '2016/9/3' '21:01'

@author: 'michael"  
"""
import numpy as np

class k_Means:

    def __init__(self,filename):
        self.datafile = filename


    def loadDataSet(self):
        dataMat = []
        fr = open(self.datafile)
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float,curLine))
            dataMat.append(fltLine)
        return dataMat

    def distEclud(self, vecA , vecB):
        return  np.sqrt(np.sum(np.power((vecA - vecB), 2)))

    def randCenter(self , dataSet, k):
        n = np.shape(dataSet)[1]   # the number of attributes in dataSet
        print(n)
        centroids = np.mat(np.zeros((k,n)))     # the average points of the k-clusters
        for j in range(n):                      # initialize the mean points of the k-clusters
            minJ = min(dataSet[:,j])
            rangJ = float(max(dataSet[:,j]) - minJ)
            centroids[:,j] = minJ + rangJ * np.random.rand(k,1)
        print(centroids)
        return centroids


    def kMeans(self, dataSet, k ):
        m = np.shape(dataSet)[0]              # the number of the vectors in dataSet
        clusterAssment =  np.mat(np.zeros((m,2)))
        print('flag')
        centroids = self.randCenter(dataSet, k)
        clusterChanged  = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                for j in range(k):

                    distJI = self.distEclud(centroids[j,:] , dataSet[i,:])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                # mark the kth label and the distance between centroid and the vector for the every single vector
                clusterAssment[i,:] = minIndex,minDist**2
            #print(centroids)
            for cent in range(k):
                ptsInClust = dataSet[np.nonzero(clusterAssment[:,0] == cent)[0]] # find all the vectors in the kth cluster
                centroids[cent,:] = np.mean(ptsInClust, axis = 0)                # calculate the mean vector of the kth cluster
        return centroids, clusterAssment


def main():
    filename = '../data/testSet.txt'
    kmeansTest = k_Means(filename)
    datMat = np.mat(kmeansTest.loadDataSet())
    kmeansTest.kMeans(datMat,4)
    centroids,clusterAssment = kmeansTest.kMeans(datMat,4)
    print(centroids)


if __name__ == '__main__':
    main()






























