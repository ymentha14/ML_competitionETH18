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
X_label = pd.read_hdf("input_data/train_labeled.h5", "train")
X_label = X_label.drop(columns = 'y')
X_label = X_label.to_numpy()
X_unlabel = pd.read_hdf("input_data/train_unlabeled.h5", "train")
X_unlabel = X_unlabel.to_numpy()
print('size of label ', len(X_label))
print('size of unlabel ',len(X_unlabel))

pca = PCA(n_components=139)
pca.fit(X_label)
plt.figure(1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

scaler = StandardScaler()
scaler.fit(X_label)
X_lab_std = scaler.transform(X_label)
pca.fit(X_lab_std)
plt.figure(2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance STD labeled data')

scaler.fit(X_unlabel)
X_unlab_std = scaler.transform(X_unlabel)
pca.fit(X_unlab_std)
plt.figure(3)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance STD unlabeled data')
score50 = np.sum(pca.explained_variance_ratio_[:50])
print('Variance explained by 50 components on the unlabeled data',score50)
plt.show()

