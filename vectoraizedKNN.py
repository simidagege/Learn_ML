#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def createDataset():
    dataset = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A','B','B']
    return dataset, labels

def KNN(trainDataset, labels, X, k):
    '''
    :param trainDataset: 训练数据集(m, n)
    :param labels: 训练集对应的标签向量(m, 1)
    :param X: 待分类数据(1, n)
    :param k: 指定相近的k数
    :return:
    '''
    rows = trainDataset.shape[0] #m
    cols = trainDataset.shape[1] #n

    #use np.tile() to remap testX into a (rows, 1) matrix
    testXMat = np.tile(testX, (rows, 1)) #(m, n)

    #get diff mat by matrix minus: x1 - x2, y1 - y2 ...
    diffMat = testXMat - trainDataset #(m, n)

    #(x1 - x2) * (x1 - x2), (y1 - y2) * (y1 - y2) ...
    squareDiffMat = np.square(diffMat)

    #(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + ...
    #donnot use keepdims = True !!!
    sumDiffMat1 = np.sum(squareDiffMat, axis = 1)

    #sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + ...)
    sqrtDiffMat = np.sqrt(sumDiffMat1)

    #sort index by iterm's ascent order
    #e.g: sqrtDiffMat = [1.3453624   1.27279221  0.14142136  0.1]
    # --> sortedDiffIndex = [3 2 1 0]
    sortedDiffIndex = sqrtDiffMat.argsort()

    classCount = {}
    for idx in range(k):
        voteLabel = labels[sortedDiffIndex[idx]]
        dist = sqrtDiffMat[sortedDiffIndex[idx]]
        if not classCount.has_key(dist):
            classCount[dist] = voteLabel
    return classCount

dataset, labels = createDataset()
#print dataset
#print dataset.shape
#print labels

testX = [0.1, 0.1]
k = 2
knearbors = KNN(dataset, labels, testX, k)
print knearbors
