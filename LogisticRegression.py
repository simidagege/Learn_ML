#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegression:
    def __init__(self, m, n, X, Y, alpha, iterStopThreshold, iterMaxCnt = 100000):
        '''
        :param m: ѵ��������
        :param n: ÿ��ѵ������������������
        :param X: ѵ����������
        :param Y: ѵ��������ǩ
        :param alpha: ѧϰ��
        :param iterStopThreshold: �����ֵ
        :param iterMaxCnt������������
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
        JOld = -1.0 # ������һ�ε����Ĵ��ۣ������´�������ж��Ƿ��������ֹͣ�������
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
            print(str(cnt) + "�ε����Ĵ���J��" + str(J))
            if np.abs(J - JOld) < self.iterStopThreshold and JOld > 0 or cnt > self.iterMaxCnt:
                break
            JOld = J
        return self.W, self.b

if __name__ == '__main__':
    m = 3  # ѵ����������
    n = 2  # ÿ������������������
    X = np.random.rand(m * n)  # �������ѵ������
    X = X.reshape(n, m)  # ����Ng�γ̣�Xת��Ϊn*m
    y = np.random.randint(0, 2, (1, m))  # 1*m
    alpha = 0.1  # ѧϰ��
    iterStopThreshold = 0.00001  # ���õ���ֹͣ�����ֵ
    iterMaxCnt = 10000 # �����������������õ���������ֹͣ����
    lr = LogisticRegression(m, n, X, y, alpha, iterStopThreshold, iterMaxCnt)
    lr.train()
    #���Է�����
    print(lr.predict(np.array([np.random.rand(1), np.random.rand(1)])))
