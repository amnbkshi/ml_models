#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:17:36 2018

@author: aman
"""

import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/BostonHousing.csv')
df.drop(['zn', 'chas'], 1, inplace=True)
df.rename(columns={'medv':'label'}, inplace=True)

X = np.array(df.drop('label', 1))
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                     y,
                                                                     test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)

#with open('linearegression.pkl', 'wb') as f:
#    pickle.dump(lr, f)
#with open('linearegression.pkl', 'rb') as f:
#    lr = pickle.load(f)

accuracy = lr.score(X_test, y_test)
predictions = lr.predict(X_test)

print(accuracy, predictions, y_test)
