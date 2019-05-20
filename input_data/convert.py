import numpy as np
import os
import pandas as pd

data_lab = pd.read_hdf("h5/train_labeled.h5", "train")
data_unlab = pd.read_hdf("h5/train_unlabeled.h5", "train")
X_submit = pd.read_hdf("h5/test.h5", "test")
path_lab = 'csv/train_labeled.csv'
path_unlab ='csv/train_unlabeled.csv'
path_test = 'csv/test.csv'
data_lab.to_csv(os.path.join(path_lab),index = False)
data_unlab.to_csv(os.path.join(path_unlab),index = False)
X_submit.to_csv(os.path.join(path_test),index = False)
