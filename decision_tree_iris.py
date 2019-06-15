#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 15:21:24 2018

@author: aman
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

#tree_reg = DecisionTreeRegressor(max_depth=2)
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)

print(clf.predict_proba([[5, 1.5]]))
print(clf.predict([[5, 1.5]]))

export_graphviz(clf,
                out_file=("iris_tree.txt"),
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True,
                filled=True
               )

clf.tree_.children_left #array of left children
clf.tree_.children_right #array of right children
clf.tree_.feature #array of nodes splitting feature
clf.tree_.threshold #array of nodes splitting points
clf.tree_.value #array of nodes values
