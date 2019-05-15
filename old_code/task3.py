#!/usr/bin/env python
# coding: utf-8

# # Task 3 version 0: 
# ## Multi-class Classification:
# 1. Your goal is to predict a discrete value y (0, 1, 2, 3 or 4) based on a vector x.
# 
# 1. Potential approaches / tools to consider: Neural networks / Deep Learning (TensorFlow, PyTorch, Theano, etc.)
# 
# 1. Note: A part of this graded task is to setup and use existing machine learning libraries. As a result, we cannot help you with this.
# 
# The training data is contained in train.h5 in the hdf5 format. Each entry in train.h5 is one data instance indexed by an Id. It consists of one integer with the class label and 120 doubles for the vector x1-x120. The test set file (test.h5) has the same structure except that the column for y is omitted.

# ### Evaluation: 
# Submissions are evaluated by accuracy, i.e., the fraction of correct predictions.
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
from tensorflow import keras
tf.set_random_seed(14)


#some useful dictionaries
losses = {'mean_squared_error':'mse','mean_absolute_error':'mae','squared_hinge':'sh','hinge':'h','sparse_categorical_crossentropy':'scc'}

p_optimizers = {'SGD':keras.optimizers.SGD,'RMSprop':keras.optimizers.RMSprop,'Adagrad':keras.optimizers.Adagrad,'Adadelta':keras.optimizers.Adadelta,'Adam':keras.optimizers.Adam,'Adamax':keras.optimizers.Adamax,'Nadam':keras.optimizers.Nadam}

activ_fctions = {'relu' :tf.nn.relu,'softmax' :tf.nn.softmax,'leak_relu':tf.nn.leaky_relu}


JSON_MODE = (len(sys.argv) >1)
PCA_MODE = False
INPUT_DIM = 120
p_pca = 30
p_decay = 0
p_momentum = 0
output_name = (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
p_batch_size =32
p_regul = 0
if (JSON_MODE):
    fn = sys.argv[1]
    if os.path.isfile(fn):
        print("successfully read the json file."+sys.argv[1])
        if ('origin' in os.path.basename(os.path.normpath(sys.argv[1]))):
            output_name += 'OR'
        json_dict = json.load(open(fn))
        RATIO = json_dict['Ratio']
        p_loss = json_dict['loss']
        p_opt = json_dict['optimizer']
        p_lr = json_dict['learning rate']
        p_metric = json_dict['metrics']
        p_epochs = json_dict['number of epochs']
        lay_node = json_dict['layers']
        PCA_MODE = ('pca' in json_dict)
        if (PCA_MODE):
            p_pca = json_dict['pca']
            INPUT_DIM = p_pca
        if ('decay' in json_dict and 'momentum' in json_dict):
            p_decay = json_dict['decay']
            p_momentum = json_dict['momentum']
        if ('batch_size' in json_dict):
            p_batch_size = json_dict['batch_size']
        if ('regul' in json_dict):
            p_regul = json_dict['regul']
    else:
        print("uncorrect path. abort.")
        print(sys.argv[1])
        #exit()
else :
    print("taking the values of the code")
    RATIO = 0.99
    p_loss = 'sparse_categorical_crossentropy'
    p_opt = 'SGD'
    p_metric = "accuracy"
    p_epochs = 100
    lay_node = [('relu',1024),('dropout',0.33),('relu',1024),('dropout',0.33),('relu',1024),('dropout',0.33),('relu',1024),('dropout',0.33),('relu',1024)]
    p_lr = 0.008
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
#import train values
train_pd = pd.read_hdf("train.h5", "train")
train_big = train_pd.to_numpy()

#change their order: NB the y values remain associated to the correct X ones. You can check it with the small example down here.
np.random.shuffle(train_big)

#with test set

[train,valid] = np.split(train_big,[round(RATIO*len(train_big))])

#split them into X and y
X_train = np.delete(train, 0, axis=1)
y_train = (train[:,0]).astype(int)

X_valid = np.delete(valid, 0, axis=1)
y_valid = (valid[:,0]).astype(int)


#process the input elements that we need to submit
X_submit = (pd.read_hdf("test.h5", "test")).to_numpy()

#PCA preprocessing
if (PCA_MODE):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_submit = scaler.transform(X_submit)
    pca = PCA(n_components=p_pca)
    X_train = pca.fit_transform(X_train)
    X_valid = pca.transform(X_valid)
    X_submit = pca.transform(X_submit)

def submission_formed(predicted_y,name ):
    result_dir = "./results"
    os.makedirs(result_dir,exist_ok=True)
    out = pd.DataFrame(predicted_y)
    out.insert(0,'Id',range(45324,len(out)+45324))
    out.rename(columns={"Id": "Id", 0: "y"},inplace = True)


    path = 'results/'+name+'.csv'
    
    out.to_csv(os.path.join(path),index = False)
#the neural network: we vary its number of layers according to the parameter lay_node

model = keras.Sequential()
for counter,(name,num) in enumerate(lay_node):
  if (counter==0):
    model.add(keras.layers.Dense(num,activation=activ_fctions[name],input_dim=INPUT_DIM))
  elif (name != 'dropout'):
    model.add(keras.layers.Dense(num,activation=activ_fctions[name],kernel_regularizer=keras.regularizers.l2(p_regul)))
  else:
    model.add(keras.layers.Dropout(num))
model.add(keras.layers.Dense(5,activation=tf.nn.softmax))

if (p_opt=='SGD'):
    optimiz = (p_optimizers[p_opt])(lr = p_lr,decay = p_decay,momentum = p_momentum)
else :
    optimiz = (p_optimizers[p_opt])(lr = p_lr,decay = p_decay)
model.compile(optimizer = optimiz,
             loss = p_loss,
             metrics=[p_metric])


#Tensorboard part
logs_base_dir = "./logs"
os.makedirs(logs_base_dir,exist_ok=True)

output_name += 'opt='+p_opt+'-lr='+str(round(p_lr,5))+'-decay='+str(p_decay)+'-loss='+ losses[p_loss]+'-epo='+str(p_epochs)+'-pca='+str(p_pca)+'regul='+str(p_regul)

output_name += '-lays=['
for counter,(name,num) in enumerate(lay_node):
  output_name +='('+str(num)+','+name[0]+')'
output_name += ']'


logdir = os.path.join(logs_base_dir,output_name)


tbCallBack = tf.keras.callbacks.TensorBoard(logdir,histogram_freq=1,write_grads=True)


model.fit(x=X_train,
        y=y_train,
        epochs = p_epochs,
        batch_size=p_batch_size,
        validation_data=(X_valid,y_valid),
        callbacks=[tbCallBack])
#if you want shuffling between each epoch
#model.fit(X_train, y_train,epochs = p_epochs, shuffle=False ,validation_split = 0.1)

test_loss, aut_acc = model.evaluate(X_valid,y_valid)

y_probas = (model.predict(X_valid))

y_valid_predicted = np.array([np.argmax(i) for i in y_probas])

man_acc = accuracy_score(y_valid_predicted, y_valid)


#Output

#predict the output
y_submit_conf = model.predict(X_submit) #for each input, we still have only the confidences, we need to take the max one
y_submit = np.array([np.argmax(i) for i in y_submit_conf])

#creates the output name
submission_formed(y_submit,output_name)

json_dict['accuracy'] = man_acc

with open(logdir+'/recap.json','w') as fp:
    json.dump(json_dict, fp, indent = 1)
