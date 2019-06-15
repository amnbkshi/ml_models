#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:32:46 2018

@author: aman
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
new_features = [5,7]

def k_nearest_neighbor(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting groups!')
        
    distances = []
    for group in data:
        for feature in data[group]:
            ed = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([ed, group])
    
    voting_groups = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(voting_groups).most_common(1)[0][0]
    confidence = Counter(voting_groups).most_common(1)[0][1] / k
    return vote_result, confidence

result, confidence = k_nearest_neighbor(dataset, new_features)
print(result, ',', confidence)


#[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0], new_features[1], s=100, color=result)
#plt.show()
