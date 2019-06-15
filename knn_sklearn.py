#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:00:02 2018

@author: aman
"""

import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

df = pd.read_csv('data/BreastCancer.csv')
df.drop('Id', 1, inplace=True)

imputer = Imputer(strategy='median')
imputer.fit(df)
df = pd.DataFrame(imputer.transform(df), columns = df.columns)

X = np.array(df.drop(['Class'], 1), dtype=np.float64)
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y)

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(accuracy)

example_features = np.array([[4,2,1,1,1,2,3,2,1]])
example_features = example_features.reshape(len(example_features), -1) # Important!
prediction = knn.predict(example_features)
print(prediction)
