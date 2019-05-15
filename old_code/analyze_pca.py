#!/usr/bin/env python
import random
import json
import numpy as np
np.random.seed(seed=14)
random.seed(14)
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import datetime
#prevent tf to use CUDA GPUs which use multithreading==> kills determinism
os.environ["CUDA_VISIBLE_DEVICE"]=""
import tensorflow as tf
import sys
#import train values
X_train = pd.read_hdf("train.h5", "train")
X_train = X_train.to_numpy()


pca = PCA(n_components=120)
pca.fit(X_train)
plt.figure(1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
scaler = StandardScaler()
scaler.fit(X_train)
X_sc_train = scaler.transform(X_train)
plt.show()
'''
pca.fit(X_sc_train)
plt.figure(2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance STD')
print("done!")
'''

