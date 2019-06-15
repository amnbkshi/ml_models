#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:14:15 2018

@author: aman
"""

import numpy as np
from sklearn.decomposition import PCA

X = [[4,5,6], [1,2,3], [7,8,9]]

pca = PCA(n_components=2) #PCA(n_components=0.95) - ratio of variance to preserve
X2D = pca.fit_transform(X)
print(X2D)

pca.components_.T[:, 0]
pca.explained_variance_ratio_ (variance lying along axis)



pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1




pca = PCA(n_components = 154)
X_mnist_reduced = pca.fit_transform(X_mnist)
X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)

