import random
import json
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import random
import json
np.random.seed(seed=14)
random.seed(14)
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICE"]=""
import sys
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation


#*****dictionnaries******
ss_models_dic = {'LabSpr':LabelSpreading(),'LabProp':LabelPropagation()}
losses = {'mean_squared_error':'mse','mean_absolute_error':'mae','squared_hinge':'sh','hinge':'h','sparse_categorical_crossentropy':'scc'}

optimizers = {'SGD':keras.optimizers.SGD,'RMSprop':keras.optimizers.RMSprop,'Adagrad':keras.optimizers.Adagrad,'Adadelta':keras.optimizers.Adadelta,'Adam':keras.optimizers.Adam,'Adamax':keras.optimizers.Adamax,'Nadam':keras.optimizers.Nadam}

activ_fctions = {'relu' :tf.nn.relu,'softmax' :tf.nn.softmax,'leak_relu':tf.nn.leaky_relu}
#************************

par_list = ['Ratio','pca','loss','optimizer','learning rate','metrics','decay','momentum','batch_size','number of epochs','layers','ss_model','ss_kernel']

###########################Default parameters#####################
#Training
RATIO = 0.7
INPUT_DIM = 139
#PCA
PCA_MODE = False
p_pca = 50

#NN
p_loss = "sparse_categorical_crossentropy"
p_opt = "SGD"
p_metric = "accuracy"
p_epochs = 5
p_batch_size = 32
p_decay = 0
p_momentum = 0
p_lr = 0.001
lay_node = ["relu",206]
#Semi Supervised 
#(1)using the sklearn
# choice between (1.1) LabelPropagation and (1.2) Labelspreading
#   (1.1)LabelPropagation: choice between knn and rbf
#   (1.2) LableSpreading : same
p_ss_mod = 'LabSpr'
p_ss_kern = 'knn'
#(2) using the paper https://github.com/tmadl/semisup-learn or https://epubs.siam.org/doi/abs/10.1137/1.9781611972795.68

###################################################################


JSON_MODE = (len(sys.argv) >1)
if (JSON_MODE):
    fn = sys.argv[1]
    if os.path.isfile(fn):
        print("successfully read the json file."+sys.argv[1])
        json_dict = json.load(open(fn))
        for i in json_dict:
            if not (i in par_list):
                print('unknown parameter. abort.')
                exit()
        PCA_MODE = ('pca' in json_dict)
        if (PCA_MODE):
            p_pca = json_dict['pca']
            INPUT_DIM = p_pca

    else:
        print("uncorrect path. abort.")
        print(sys.argv[1])
        exit()
else :
    print("taking the values of the code")
    json_dict = {
            'Ratio':RATIO,
            'loss':p_loss,
            'optimizer':p_opt,
            'metrics':p_metric,
            'number of epochs': p_epochs,
            'batch_size':p_batch_size,
            'decay':p_decay,
            'momentum':p_momentum,
            'learning rate' :p_lr,
            'layers':lay_node,
            }
    if (PCA_MODE):
        json_dict['pca'] = p_pca
        INPUT_DIM = p_pca

#import train values
data_lab = pd.read_hdf("input_data/train_labeled.h5", "train").to_numpy()
X_big_lab = data_lab[:,1:]
y_big = data_lab[:,0]
X_train_lab, X_valid_lab,y_train,y_valid = train_test_split(X_big_lab,y_big,test_size=0.33,random_state=14)
X_unlab = (pd.read_hdf("input_data/train_unlabeled.h5", "train")).to_numpy()
X_tot = np.concatenate((X_train_lab,X_unlab),axis=0)
X_submit = (pd.read_hdf("input_data/test.h5", "test")).to_numpy()

#train_unlab = train_unlab_pd.to_numpy()


#PCA preprocessing
if (PCA_MODE):
    scaler = StandardScaler()
    X_tot = scaler.fit_transform(X_tot)
    X_train_lab = scaler.transform(X_train_lab)
    X_unlab = scaler.transform(X_unlab)
    X_valid_lab = scaler.transform(X_valid_lab)
    X_submit = scaler.transform(X_submit)
    
    pca = PCA(n_components=p_pca)
    pca.fit(X_tot)
    X_train_lab = pca.transform(X_train_lab)
    X_unlab = pca.transform(X_unlab)
    X_valid_lab = pca.transform(X_valid_lab)
    X_submit = pca.transform(X_submit)

#Semi supervised algo

