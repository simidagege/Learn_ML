#!/usr/bin/env python2
#-*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import sys
#error metrics
import sklearn.metrics as sm

#filename = sys.argv[1]
filename = "data_singlevar.txt"
X = []
y = []

#load dataset
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [ float(i) for i in line.split(',') ]
        X.append(xt)
        y.append(yt)
        #print xt, yt

#split dataset to training dataset(80 percent) and testing dataset(20 percent)
num_train = int(0.8 * len(X))
num_test = len(X) - num_train

X_train = np.array(X[:num_train]).reshape((num_train, 1))
y_train = np.array(y[:num_train])

X_test = np.array(X[num_train:]).reshape((num_test, 1))
y_test = np.array(y[num_train:])

from sklearn import linear_model

#create linear regression model
linear_regressor = linear_model.LinearRegression()

#train model
linear_regressor.fit(X_train, y_train)

'''
import matplotlib.pyplot as plt

#plot trained model for train dataset
y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color = 'green')
plt.scatter(X_train, y_train_pred, color = 'red')
plt.plot(X_train, y_train_pred, color = 'black', linewidth = 4)
plt.title('Traing data')
plt.show()

#plot predicts for test dataset
y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test, y_test, color = 'green')
plt.scatter(X_test, y_test_pred, color = 'red')
plt.plot(X_test, y_test_pred, color = 'black', linewidth = 4)
plt.title('Testing data')
plt.show()



#mean absolute error
print "Mean absolute error =", round(sm.mean_absolute_error(y_test_pred, y_test), 2)
#mean squared error
print "Mean squared error =", round(sm.mean_squared_error(y_test_pred, y_test), 2)
#median absolute error
print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)
#explained variance score
print "Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)
#r2 score
print "R2 score =", round(sm.r2_score(y_test, y_test_pred), 2)

#save/load model
import cPickle as pickle

#save model
output_model_file = "saved_model.pkl"
with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)

#load model
with open(output_model_file, 'r') as f:
    model_linreger = pickle.load(f)

y_test_pred_new = model_linreger.predict(X_test)
print "Mean squared error =", round(sm.mean_squared_error(y_test_pred_new, y_test), 2)
'''

#ridge regression
ridge_regressor = linear_model.Ridge(alpha=0.01,max_iter=10000)

#train
ridge_regressor.fit(X_train, y_train)
#predict
y_test_pred_ridge = ridge_regressor.predict(X_test)
print "Ridge regression model's mean squared error =", round(sm.mean_squared_error(y_test_pred_ridge, y_test), 2)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures

#create PolynomialFeatures with degree = 3: x ^ 3
polynomial = PolynomialFeatures(degree=3)

#transform X_train to polynomial features
x_train_transformed = polynomial.fit_transform(X_train)
x_test_transformed = polynomial.fit_transform(X_test)

#train model use polynomial features x_train_transformed
polyLinearModel = linear_model.LinearRegression()

#fit model
polyLinearModel.fit(x_train_transformed, y_train)

#test model by test dataset
y_test_pred_poly = polyLinearModel.predict(x_test_transformed)
print "Polynomial regression model's mean squared error =", round(sm.mean_squared_error(y_test_pred_poly, y_test), 2)
