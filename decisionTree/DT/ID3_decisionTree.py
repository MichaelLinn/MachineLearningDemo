# -*- coding: utf-8 -*-

"""
Created on Thu Jul 28 12:36:46 2016
@author: michael
"""

from math import log

class ID3_decisionTree:

    def __init__(self, dataSet,labels):
        self.dataSet = dataSet
        self.labels = labels
    # 基于特征值A_j，把D分成D_1,D_2,...,D_i,...,D_n, 此处计算 H(D_i|A_j = a_i)
    # empirical entropy = sum_i( p_i * H(D|A_j = a_i) )
    #                   = sum_i ( |D_i|/|D| * sum_k( - (|D_ik| / |D_i|) log (|D_ik| / |D_i|) ))

    def calcShannonEnt(self, dataSet): # calculate every class's Shannon Entropy
        numEntries = len(dataSet)   # the number of the vectors
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1] #  the last line of the data set is the class labels
            labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1  # calculate the number of the class labels
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key] / numEntries)
            shannonEnt += -(prob * log(prob, 2))
        return shannonEnt

    # 在数据集D中去掉特征值A_i，方便计算基于属性A_i划分后的数据集D_n的熵

    # delete specific feature value in all the vector by traversing the data set
    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reduceFeatVec)
        return retDataSet

    #  choose the feature which has the most information gain
    def chooseBestFeatureToSplit(self, dataSet):
        #   the number of the feature's types
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(self.dataSet)  # H(D)
        baseInfoGain = 0.0
        bestFeature = -1

        for i in range(numFeatures):
            #       empirical entropy = sum_i( p_i * H(D|A = a_i) )
            #       计算  H(D_i|A = a_i)
            featList = [example[i] for example in dataSet]
            # 用集合的形式存储属性A的所有取值，重复的值只保存一次
            # uniqueVas == D_i 的组数    （a_i数量是对应D_i数量的）
            uniqueVals = set(featList)
            empiricalEntropy = 0.0
            for featrue in uniqueVals:
                subDataSet = self.splitDataSet(dataSet,i, featrue)
                prob = len(subDataSet) / float(len(dataSet))
                empiricalEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - empiricalEntropy
            if (infoGain > baseInfoGain):
                baseInfoGain = infoGain
                bestFeature = i
        return bestFeature

    #   投票选出占多数的类
    def majorityClass(self, classList):
        classCount = {}
        for vote in classList:
            classCount[vote] = classCount.get(classCount[vote], 0) + 1
        # 字典类型排序后转换成list类型
        sortedClassList = sorted(classCount.items(), key=lambda classCount: classCount[1], reverse=True)
        return sortedClassList[0][0]

    #   labels是所有属性值的标签列表
    def createDecisionTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]    # class label vector
        # 当标签列表中所有的 类别 都相同后，不需要继续分类  递归结束
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 当所有属性都被遍历过后(只剩下标签，所以len(dataSet[0] == 1)，递归结束   通过投票来判断最后的类被归属
        if len(dataSet[0]) == 1:     # the length of (data[0]-1) is equal to the number of the features
            return self.majorityClass(classList)  # vote for the best classification (let the major class be the class)
        # 选出当前dataSet中拥有最大information gain的属性值
        bestFeature = self.chooseBestFeatureToSplit(dataSet)
        # 当前最佳划分属性值的标签名
        bestLabel = labels[bestFeature]
        # 用字典来存储树
        decisionTree = {bestLabel: {}}
        #在属性标签列表中删除当前最佳属性
        del (labels[bestFeature])
        #遍历当前最佳划分属性的所有取值，即求出当前树有多少分支
        featValues = [example[bestFeature] for example in dataSet]
        uniqueVals = set(featValues)
        #当前最佳属性有多少取值，当前的决策树就有多少个分支，通过递归为每个分支生成子树
        for value in uniqueVals:
            sublabels = labels[:]
            #splitDataSet()方法  得到当前树其中一个分支的dataSet（即删除当前dataSet中最佳属性整列并 选出满足dataSet.bestFeature == value的所有行 组成新的dataSet）
            decisionTree[bestLabel][value] = self.createDecisionTree(self.splitDataSet(dataSet, bestFeature, value),sublabels)
        return decisionTree


def main():
    print("the test is running!")
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ["no surfacing" , "flippers"]
#   dt = ID3_decisionTree(dataSet)
#   print(dt.calcShannonEnt())
#   id3_DT = dt.createDecisionTree(dataSet,labels)
#   print(id3_DT)

    fr = open("../trainingData/lenses.txt")
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    lenseLabels = ['age','prescripe','astigmatic','tearRate']
    dt = ID3_decisionTree(lenses,lenseLabels)
    id3_DT = dt.createDecisionTree(dt.dataSet,dt.labels)
    print(id3_DT)

if __name__ == '__main__':
    main()

# result
# {'tearRate': {'reduced': 'no lenses', 'normal':
# {'astigmatic': {'yes': {'prescripe': {'myope': 'hard', 'hyper':
# {'age': {'pre': 'no lenses', 'presbyopic': 'no lenses', 'young': 'hard'}}}}, 'no':
# {'age': {'pre': 'soft', 'presbyopic': {'prescripe': {'myope': 'no lenses', 'hyper': 'soft'}}, 'young': 'soft'}}}}}}
