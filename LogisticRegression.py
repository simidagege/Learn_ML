#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegression:
    def __init__(self, m, n, X, Y, alpha, iterStopThreshold, iterMaxCnt = 100000):
        '''
        :param m: 训练样本数
        :param n: 每个训练样本包含的特征数
        :param X: 训练样本矩阵
        :param Y: 训练样本标签
        :param alpha: 学习率
        :param iterStopThreshold: 误差阈值
        :param iterMaxCnt：最大迭代次数
        '''

        self.X = X
        self.Y = Y
        self.m = m
        self.n = n
        self.W = np.zeros((n, 1))
        self.b = 0.0
        self.alpha = alpha
        self.iterStopThreshold = iterStopThreshold
        self.iterMaxCnt = iterMaxCnt

    def predict(self, x):
        result = np.dot(self.W.T, x) + self.b
        result = 1 / (1 + np.exp(-result))
        if result > 0.5:
            return 1
        else:
            return 0

    def train(self):
        cnt = 0
        JOld = -1.0 # 保存上一次迭代的代价，与最新代价相减判断是否满足迭代停止误差条件
        while True:
            cnt += 1
            Z = np.dot(self.W.T, self.X) + self.b
            A = 1 / (1 + np.exp(-Z))
            J = -( np.dot(self.Y, np.log(A).T) + np.dot(1 - self.Y, np.log(1 - A).T) )
            dz = A - self.Y
            dw = np.dot(self.X, dz.T) / self.m
            db = np.sum(dz, axis = 1) / self.m
            J /= self.m
            self.W -= self.alpha * dw
            self.b -= self.alpha * db
            print(str(cnt) + "次迭代的代价J：" + str(J))
            if np.abs(J - JOld) < self.iterStopThreshold and JOld > 0 or cnt > self.iterMaxCnt:
                break
            JOld = J
        return self.W, self.b

if __name__ == '__main__':
    m = 3  # 训练样本个数
    n = 2  # 每个样本包含的特征数
    X = np.random.rand(m * n)  # 生成随机训练样本
    X = X.reshape(n, m)  # 基于Ng课程，X转换为n*m
    y = np.random.randint(0, 2, (1, m))  # 1*m
    alpha = 0.1  # 学习率
    iterStopThreshold = 0.00001  # 设置迭代停止误差阈值
    iterMaxCnt = 10000 # 最大迭代次数，超过该迭代次数则停止迭代
    lr = LogisticRegression(m, n, X, y, alpha, iterStopThreshold, iterMaxCnt)
    lr.train()
    #测试分类结果
    print(lr.predict(np.array([np.random.rand(1), np.random.rand(1)])))
