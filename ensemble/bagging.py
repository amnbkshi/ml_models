#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 17:23:11 2018

@author: aman
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

data = make_moons()
X = data[0]
y = data[1]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y)


bag_clf = BaggingClassifier(
                            DecisionTreeClassifier(),
                            n_estimators=100,
                            max_samples=0.1,
                            bootstrap=True,
                            n_jobs=-1,
                            oob_score=True,
                            verbose=3
                            )

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(bag_clf.oob_score_)
#clf.oob_decision_function_

#max_features and bootstrap_features: use these two features
#for random patches and random subspaces


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

#EXTRA TREES
ExtraTreesClassifier()

#Feature selection using RF
#important features appear near root and unimportant at leaves
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
