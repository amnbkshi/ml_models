#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:57:05 2018

@author: aman
"""

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                             n_estimators=200,algorithm="SAMME.R", 
                             learning_rate=0.5)
ada_clf.fit(X_train, y_train)


from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2,
                                 n_estimators=3,
                                 learning_rate=1.0)
gbrt.fit(X, y)


