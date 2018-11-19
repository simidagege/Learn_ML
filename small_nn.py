#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

class SmallNeuralNetwork(object):
    def __init__(self, m, n, X, Y, hln, alpha, iterStopThreshold, iterMaxCnt = 100000):
        '''
        :param m: 训练样本数
        :param n: 每个训练样本包含的特征数
        :param X: 训练样本矩阵
        :param Y: 训练样本标签
        :param hln(hidden layer nodes): 隐藏层的神经元个数
        :param alpha: 学习率
        :param iterStopThreshold: 误差阈值
        :param iterMaxCnt：最大迭代次数
        '''

        self.X = X #dims: (n, m)
        self.Y = Y #dims: (1, m)
        self.m = m
        self.n = n
        self.hln =hln #hidden layer nodes
        self.alpha = alpha
        self.iterStopThreshold = iterStopThreshold
        self.iterMaxCnt = iterMaxCnt

        #hidden layer paramters
        #dims of W1: (hln, n)
        #dims of b1: (hln, 1)
        self.W1 = np.random.randn(hln, n) * 0.01
        self.b1 = np.zeros((hln, 1))

        #output layer paramters
        #dims of W2: (hln, n)
        #dims of b2: (1, 1)
        self.W2 = np.random.randn(1, hln) * 0.01
        self.b2 = 0.0

    def predict(self, x):
        Z1 = np.dot(self.W1, x) + self.b1
        A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = 1 / (1 + np.exp(-Z2))
        if A2 >= 0.5:
            return 1
        else:
            return 0

    def train(self):
        Jold = -1.0 # old cost
        iterCnt = 0 # iterator count
        while True:
            #-------update iterator count
            iterCnt += 1

            #---------forward propagation-------------------
            # dims of Z1: (hln, m) = (hln, n) * (n, m) + (hln, 1)
            Z1 = np.dot(self.W1, self.X) + self.b1
            # dims of A1: (hln, m)
            A1 = 1 / (1 + np.exp(-Z1))
            # dims of Z2: (1, m) = (1, hln) * (hln, m) + (1, 1)
            Z2 = np.dot(self.W2, A1) + self.b2
            # dims of A2: (1, m)
            A2 = 1 / (1 + np.exp(-Z2))
            # dims of Jnew: (1, 1) = (1, m) * (m, 1) + (1, m) * (m, 1)
            Jnew = -(np.dot(self.Y, np.log(A2).T) + np.dot(1 - self.Y, np.log(1 - A2).T))
            Jnew /= self.m

            #--------backward propagation----------------
            #update output layer paramters
            # dims of dz2: (1, m) = (1, m) - (1, m)
            dz2 = A2 - self.Y
            # dims of dw2: (1, hln) = (1, m) * (m, hln)
            dw2 = np.dot(dz2, A1.T) / self.m
            # dims of db2: (1, 1)
            db2 = np.sum(dz2, axis = 1, keepdims = True) / self.m
            self.W2 -= self.alpha * dw2
            self.b2 -= self.alpha * db2

            #update hidden layer paramters
            # dims of dz1: (hln, m) = ((hln, 1) * (1, m)) * ((m, hln) * (hln, m))
            #                       = ((hln, m) * (m, m))
            #                       = (hln, m)
            dz1 = np.dot(np.dot(self.W2.T, dz2), np.dot(Z1.T, 1 - Z1))
            # dims of dw1: (hln, n) = (hln, m) * (m, n)
            dw1 = np.dot(dz1, self.X.T) / self.m
            # dims of db1: (hln, 1)
            db1 = np.sum(dz1, axis = 1, keepdims = True)
            self.W1 -= self.alpha * dw1
            self.b1 -= self.alpha * db1

            #-----judge to stop iteration-----
            print("第" + str(iterCnt) + "次迭代的代价J：" + str(Jnew))
            if np.abs(Jnew - Jold) < self.iterStopThreshold and Jold > 0.0:
                print "Jnew - Jold(%s) < %s, stop iteration." % (abs(Jnew - Jold), self.iterStopThreshold)
                break
            elif iterCnt > self.iterMaxCnt:
                print "iterCnt(%s) > iterMaxCnt(%s), stop iteration." % (iterCnt, self.iterMaxCnt)
                break

            #-----update cost-----
            Jold = Jnew

        return self.W1, self.b1, self.W2, self.b2


if __name__ == '__main__':
    '''
    m = 100  # 训练样本个数
    n = 2  # 每个样本包含的特征数
    X = np.random.rand(m, n)  # 生成随机训练样本
    X = X.reshape(n, m)  # 基于Ng课程，X转换为n*m
    y = np.random.randint(0, 2, (1, m))  # 1*m
    hidden_layer_nodes = 10
    alpha = 0.1  # 学习率
    iterStopThreshold = 0.001  # 设置迭代停止误差阈值
    iterMaxCnt = 10000 # 最大迭代次数，超过该迭代次数则停止迭代
    lr = SmallNeuralNetwork(m, n, X, y, hidden_layer_nodes, alpha, iterStopThreshold, iterMaxCnt)
    lr.train()
    #测试分类结果
    print(lr.predict(np.array([np.random.rand(1), np.random.rand(1)])))
    '''

    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    ### START CODE HERE ### (choose your dataset)
    dataset = "noisy_moons"
    ### END CODE HERE ###

    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    plt.show()

    hidden_layer_nodes = 10
    alpha = 0.01  # 学习率
    iterStopThreshold = 0.00001  # 设置迭代停止误差阈值
    iterMaxCnt = 10000  # 最大迭代次数，超过该迭代次数则停止迭代
    lr = SmallNeuralNetwork(X.shape[1], X.shape[0], X, Y, hidden_layer_nodes, alpha, iterStopThreshold, iterMaxCnt)
    lr.train()

