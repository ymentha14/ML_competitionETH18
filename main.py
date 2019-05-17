import random
import json
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
import tensorflow as tf
import sys
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
tf.set_random_seed(14)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation


#*****dictionnaries******
#allows to keep a small in the json as parameter so that it doesnt overload the filename, and still being able to link it to a real function
dic_ss_mod = {'LabSpr':LabelSpreading,'LabProp':LabelPropagation}

losses = {'mean_squared_error':'mse','mean_absolute_error':'mae','squared_hinge':'sh','hinge':'h','sparse_categorical_crossentropy':'scc'}

dic_opt = {'SGD':keras.optimizers.SGD,'RMSprop':keras.optimizers.RMSprop,'Adagrad':keras.optimizers.Adagrad,'Adadelta':keras.optimizers.Adadelta,'Adam':keras.optimizers.Adam,'Adamax':keras.optimizers.Adamax,'Nadam':keras.optimizers.Nadam}

dic_activ_fctions = {'relu' :tf.nn.relu,'softmax' :tf.nn.softmax,'leak_relu':tf.nn.leaky_relu}

#************************

#list of all potential parameters
params_nn = ['loss','optimizer','learning rate','metrics','decay','momentum','batch_size','number of epochs','layers','patience']
params_ss = ['ss_model','ss_kernel','gamma','neighbor','alpha']
param_list = ['Ratio','pca','UsingNN','paramsout','UsingSS']+params_nn+params_ss
param_out = ['Ratio','pca','optimizer','layers']

###########################Default parameters#####################
#NB:if some mandatory parameters are lacking in the json, default values will be taken

#Training
RATIO = 0.7
INPUT_DIM = 139

#PCA
PCA_MODE = False
p_pca = 50

#Early stopping
EARLY_STOP_MODE = False
p_patience = 50

#NN
USING_NN = False
p_loss = "sparse_categorical_crossentropy"
p_opt = "SGD"
p_lr = 0.001
p_metric = "accuracy"
p_decay = 0
p_momentum = 0
p_batch_size = 32
p_epochs = 5
lay_node = [("relu",206),('dropout',0.33)]

#Semi Supervised 
USING_SS = True
#(1)using the sklearn
# choice between (1.1) LabelPropagation and (1.2) Labelspreading
#   (1.1)LabelPropagation: choice between knn and rbf
#   (1.2) LableSpreading : same
p_ss_mod = 'LabSpr'
p_ss_kern = 'knn'
p_gamma = 20
p_neighbors = 7
p_alpha = 0.2
#(2) using the paper https://github.com/tmadl/semisup-learn or https://epubs.siam.org/doi/abs/10.1137/1.9781611972795.68

###################################################################

def check(inner,outer):
    for i in inner:
        if not (i in outer): 
            print('unknown parameter. abort.',i)
            exit()

JSON_MODE = (len(sys.argv) >1)
if (JSON_MODE):
    fn = sys.argv[1]
    if os.path.isfile(fn):
        print("successfully read the json file."+sys.argv[1])
        json_dict = json.load(open(fn))
        assert 'UsingNN' in json_dict and 'UsingSS' in json_dict and 'paramsout' in json_dict
        USING_NN = json_dict['UsingNN']
        USING_SS = json_dict['UsingSS']
        check(json_dict,param_list)
        check(json_dict['paramsout'],param_list)
        #iterate over the printed parameters and ensure they exist
        param_out = json_dict['paramsout']
        RATIO = json_dict['Ratio']
        if (USING_SS):
            p_ss_mod = json_dict['ss_model']
            p_ss_kern = json_dict['ss_kernel']
            p_gamma = json_dict['gamma']
            p_neighbors = json_dict['neighbor']
            p_alpha = json_dict['alpha']
        if (USING_NN):
            p_loss = json_dict['loss']
            p_opt = json_dict['optimizer']
            p_lr = json_dict['learning rate']
            p_metric = json_dict['metrics']
            p_decay = json_dict['decay']
            p_momentum = json_dict['momentum']
            p_batch_size = json_dict['batch_size']
            p_epochs = json_dict['number of epochs']
            lay_node = json_dict['layers']
        PCA_MODE = ('pca' in json_dict)
        if (PCA_MODE):
            p_pca = json_dict['pca']
            INPUT_DIM = p_pca
        EARLY_STOP_MODE = ('patience' in json_dict)
        if (EARLY_STOP_MODE):
            p_patience = json_dict['patience']
    else:
        print("uncorrect path. abort.")
        print(sys.argv[1])
        exit()
else :
    print("taking the values of the code")
    json_dict = {
            'Ratio':RATIO,
            'ss_model':p_ss_mod,
            'ss_kernel':p_ss_kern,
            'loss':p_loss,
            'optimizer':p_opt,
            'learning rate' :p_lr,
            'metrics':p_metric,
            'decay':p_decay,
            'momentum':p_momentum,
            'batch_size':p_batch_size,
            'number of epochs': p_epochs,
            'gamma':p_gamma,
            'neighbor':p_neighbors,
            'alpha':p_alpha,
            'layers':lay_node,
            }
    if (PCA_MODE):
        json_dict['pca'] = p_pca
        INPUT_DIM = p_pca
    if (EARLY_STOP_MODE):
        jsondict['patience'] = p_patience

#Adapt the parameters which need the dictionnary


#import train values
data_lab = pd.read_hdf("input_data/train_labeled.h5", "train")

X_big_lab = (data_lab.to_numpy())[:,1:]

y_big = ((data_lab.to_numpy())[:,0]).astype(int)


X_train_lab, X_valid_lab,y_train,y_valid = train_test_split(X_big_lab,y_big,test_size=0.33,random_state=14)

data_unlab = pd.read_hdf("input_data/train_unlabeled.h5", "train")

X_unlab = data_unlab.to_numpy()

X_tot = np.concatenate((X_train_lab,X_unlab),axis=0)

y_tot = np.concatenate((y_train,np.full(len(X_unlab),-1)))

X_submit = (pd.read_hdf("input_data/test.h5", "test")).to_numpy()

def pca_preprocess():
    global X_tot, X_train_lab, X_unlab, X_valid_lab, X_submit
    scaler = StandardScaler()
    X_tot = scaler.fit_transform(X_tot)
    X_train_lab = scaler.transform(X_train_lab)
    X_unlab = scaler.transform(X_unlab)
    X_valid_lab = scaler.transform(X_valid_lab)
    X_submit = scaler.transform(X_submit)
    
    pca = PCA(n_components=p_pca)
    X_tot = pca.fit_transform(X_tot)
    X_train_lab = pca.transform(X_train_lab)
    X_unlab = pca.transform(X_unlab)
    X_valid_lab = pca.transform(X_valid_lab)
    X_submit = pca.transform(X_submit)


def submission_formed(predicted_y,name ):
    result_dir = "./results"
    os.makedirs(result_dir,exist_ok=True)
    out = pd.DataFrame(predicted_y)
    out.insert(0,'Id',range(30000,len(out)+30000))
    out.rename(columns={"Id": "Id", 0: "y"},inplace = True)
    path = 'results/'+name+'.csv'
    out.to_csv(os.path.join(path),index = False)
#Build model
def build_model():
    model = keras.Sequential()
    for counter,(name,num) in enumerate(lay_node):
      if (counter==0):
        model.add(keras.layers.Dense(num,activation=dic_activ_fctions[name],input_dim=INPUT_DIM))
      elif (name != 'dropout'):
        model.add(keras.layers.Dense(num,activation=dic_activ_fctions[name]))
      else:
        model.add(keras.layers.Dropout(num))
    model.add(keras.layers.Dense(10,activation=tf.nn.softmax))
    #optimizer
    if (p_opt=='SGD'):
        optimiz = (dic_opt[p_opt])(lr = p_lr,decay = p_decay,momentum = p_momentum)
    else :
        optimiz = (dic_opt[p_opt])(lr = p_lr,decay = p_decay)
    model.compile(optimizer = optimiz,
                 loss = p_loss,
                 metrics=[p_metric])
    return model

def build_output_name():
    output_name = (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if (JSON_MODE):
        if ('origin' in os.path.basename(os.path.normpath(sys.argv[1]))):
            output_name+='_OR_'
        nn_string = 'NN:'
        ss_string = 'SS:'
        for i in param_out:
            temp = (i + '=' + str(json_dict[i]))
            if ( i in params_nn):
                nn_string += temp     
            elif ( i in params_ss):
                ss_string += temp 
            else:
                output_name += temp
        if (USING_SS):
            output_name += ss_string
        if (USING_NN):
            output_name += nn_string
    return output_name
########################################STARTOFCODE##########################
#Tensorboard part
logs_base_dir = "./logs"
os.makedirs(logs_base_dir,exist_ok=True)
#Build 
output_name = build_output_name()

log_spec = os.path.join(logs_base_dir,output_name)

os.makedirs(log_spec,exist_ok=True)

#PCA preprocessing
if (PCA_MODE):
    pca_preprocess()

#Semi supervised algo
if (USING_SS):
    if (p_ss_mod=='LabSpr'):
            label_prop_model = dic_ss_mod[p_ss_mod](kernel=p_ss_kern,gamma=p_gamma,n_neighbors=p_neighbors,alpha=p_alpha)
    else:            
        label_prop_model = dic_ss_mod[p_ss_mod](kernel=p_ss_kern,gamma=p_gamma,n_neighbors=p_neighbors)
    print('Start to fit. Run for shelter!')
    label_prop_model.fit(X_tot,y_tot)
    RESULT_ACC = label_prop_model.score(X_valid_lab,y_valid)
    print('accuracy obtained on the test set:',RESULT_ACC)
    y_submit = label_prop_model.predict(X_submit)

if (USING_NN):
    model = build_model()

    call_back_list = [] 

    call_back_list.append(ModelCheckpoint(log_spec+'/best_mod.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True))

    call_back_list.append(tf.keras.callbacks.TensorBoard(log_spec,histogram_freq=1,write_grads=True))

    if (EARLY_STOP_MODE):
        call_back_list.append(EarlyStopping( patience=p_patience, verbose=1, mode='min'))

    model.fit(x=X_train_lab,
            y=y_train,
            epochs = p_epochs,
            batch_size=p_batch_size,
            validation_data=(X_valid_lab,y_valid),
            callbacks=call_back_list)
    #if you want shuffling between each epoch
    #model.fit(X_train, y_train,epochs = p_epochs, shuffle=False ,validation_split = 0.1)

    test_loss, aut_acc = model.evaluate(X_valid_lab,y_valid)

    y_probas = (model.predict(X_valid_lab))

    y_valid_predicted = np.array([np.argmax(i) for i in y_probas])

    RESULT_ACC = accuracy_score(y_valid_predicted, y_valid)


    #Output

    #predict the output
    y_submit_conf = model.predict(X_submit) #for each input, we still have only the confidences, we need to take the max one
    y_submit = np.array([np.argmax(i) for i in y_submit_conf])

#creates the output name
submission_formed(y_submit,output_name)

json_dict['accuracy'] = RESULT_ACC

with open(log_spec+'/recap.json','w') as fp:
    json.dump(json_dict, fp, indent = 1)
