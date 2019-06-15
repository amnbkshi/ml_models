#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 20:06:39 2018

@author: aman
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.cluster import KMeans

df = pd.read_excel('data/titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
encoder = LabelEncoder()
 
columns = df.columns

for column in columns:
    if df[column].dtype != np.int64 and df[column].dtype != np.float64:
        df.fillna(str(0), inplace=True)
        df[column] = list(map(str, df[column]))
    else:
        df.fillna(0, inplace=True)

for column in columns:
    if df[column].dtype != np.int64 and df[column].dtype != np.float64:
        new_col = encoder.fit_transform(df[column])
        df[column] = new_col

X = np.array(df.drop(['survived'], 1)).astype(float)
X = scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)
#clf.cluster_centers_
#clf.labels_

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
        
print(correct/len(X))
