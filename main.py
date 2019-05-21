import random
import json
import numpy as np
#np.random.seed(seed=14)
#random.seed(14)
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICE"]=""
import sys
import tensorflow as tf
import sys
#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow import keras
import keras
#from tensorflow import keras
#tf.set_random_seed(14)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation


#*******************************dictionnaries*****************************
#allows to keep a small in the json as parameter so that it doesnt overload the filename, and still being able to link it to a real function
dic_ss_mod = {'LabSpr':LabelSpreading,'LabProp':LabelPropagation}

losses = {'mean_squared_error':'mse','mean_absolute_error':'mae','squared_hinge':'sh','hinge':'h','sparse_categorical_crossentropy':'scc'}

dic_opt = {'SGD':keras.optimizers.SGD,'RMSprop':keras.optimizers.RMSprop,'Adagrad':keras.optimizers.Adagrad,'Adadelta':keras.optimizers.Adadelta,'Adam':keras.optimizers.Adam,'Adamax':keras.optimizers.Adamax,'Nadam':keras.optimizers.Nadam}

dic_activ_fctions = {'relu' :tf.nn.relu,'softmax' :tf.nn.softmax}#,'leak_relu':tf.nn.leaky_relu}

#************************************************************************

#list of all potential parameters
params_nn = ['loss','optimizer','learning rate','metrics','decay','momentum','batch_size','number of epochs','layers','patience']
#loss:              loss used for the NN, cf the dictionnary above
#optimizer:         Adam,SGD etc, cf the dictionnary above
#learning rate  
#metrics            accuracy, we wont change it normally
#decay:             decay of the learning rate, generally of the order 1e-5
#momentum:          momentum of the lr
#patience:          number of epochs you wait if you use earlystopmode for the validation accuracy to increase again
#layers:            shape of the network

params_ss = ['manyfit','ss_model','ss_kernel','gamma','neighbor','alpha']
#manyfit:           since the ss accuracy has some variance but doesnt take much to be computed, manyfit designs how many independant times we run it before averaging it in order to obtain a better estimation of the accuracy in question
#ss_model:          'LabSpr' or 'LabProp'. So far, only LabSpr has converged
#ss_kernel:         'knn' or 'rbf. So far only knn converges. ***WATCH OUT***: when using rbf, euler will complain that you use too much memory!!
#gamma              parameter for the rbf
#neighbor           parameter for knn
#alpha              parameter for knn and rbf: tells at which point you will take the information of your neighbors into account

param_list = ['Ratio','pca','UsingNN','paramsout','data_state','scaler']+params_nn+params_ss
#Ratio              ratio represented by the training set
#pca                number of principal components to use. if not present, no pca will be done
#UsingNN            if set to false, the NN is not used.
#data_state         'save' or 'load'. If you want to train the NN only without having to run the ss algo again, do one run with data_state to true, and use data_state= 'load for the next ones.
#scaler             'normal' or 'standard' describes the preprocessing before applying the pca
#paramsout:         designates which parameters will be present in the output name ==> put the one you're playing with in order to easily see the difference
param_out = ['Ratio','pca','optimizer','layers']


###########################Default parameters#####################
#NB:if some mandatory parameters are lacking in the json, default values will be taken

#Training
RATIO = 0.7
INPUT_DIM = 139

#PCA
p_scaler = 'Standard'
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
p_datastate = 'save'
p_ss_mod = 'LabSpr'
p_ss_kern = 'knn'
p_gamma = 20
p_neighbors = 7
p_alpha = 0.2
p_manyfit = 1
###################################################################
#(1)Extracting the data
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
        assert 'UsingNN' and 'paramsout' in json_dict
        USING_NN = json_dict['UsingNN']
        check(json_dict,param_list)
        check(json_dict['paramsout'],param_list)
        #iterate over the printed parameters and ensure they exist
        param_out = json_dict['paramsout']
        RATIO = json_dict['Ratio']
        p_ss_mod = json_dict['ss_model']
        p_ss_kern = json_dict['ss_kernel']
        p_gamma = json_dict['gamma']
        p_neighbors = json_dict['neighbor']
        p_alpha = json_dict['alpha']
        p_datastate = json_dict['data_state']
        p_scaler = json_dict['scaler']
        if ('manyfit' in json_dict):
          p_manyfit = json_dict['manyfit']
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
            'manyfit':p_manyfit,
            'scaler':p_scaler
            }
    if (PCA_MODE):
        json_dict['pca'] = p_pca
        INPUT_DIM = p_pca
    if (EARLY_STOP_MODE):
        jsondict['patience'] = p_patience

def init_variables():
  global X_submit, X_big_lab, y_big, X_train_lab, X_valid_lab, y_train,y_valid,X_unlab,X_tot,y_tot
  X_submit = X_sub_pd.to_numpy()
  X_big_lab = (data_lab.to_numpy())[:,1:]
  y_big = ((data_lab.to_numpy())[:,0]).astype(int)
  X_train_lab, X_valid_lab,y_train,y_valid = train_test_split(X_big_lab,y_big,test_size = (1-RATIO))#,random_state=14)
  X_unlab = data_unlab.to_numpy()
  X_tot = np.concatenate((X_train_lab,X_unlab),axis=0)
  y_tot = np.concatenate((y_train,np.full(len(X_unlab),-1)))

def pca_preprocess():
    global X_tot, X_train_lab, X_unlab, X_valid_lab, X_submit
    if (p_scaler == 'Standard'):
      scaler = StandardScaler()
    elif (p_scaler == 'Normal'):
      scaler = Normalizer()
    else:
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
        output_name += ss_string
        if (USING_NN):
            output_name += nn_string
    return output_name

def submission_formed(predicted_y,name ):
    result_dir = "./results"
    os.makedirs(result_dir,exist_ok=True)
    out = pd.DataFrame(predicted_y)
    out.insert(0,'Id',range(30000,len(out)+30000))
    out.rename(columns={"Id": "Id", 0: "y"},inplace = True)
    path = 'results/'+name+'.csv'
    out.to_csv(os.path.join(path),index = False)

#useful when p_datastate is set to 'save': save the datas obtained after the ss algo
def save_to_csv(X_tot,y_tot):
    out_x = pd.DataFrame(X_tot)
    out_y = pd.DataFrame(y_tot) 
    os.makedirs('./saved_datas',exist_ok=True)
    path_x = 'saved_datas/X_tot.csv'
    path_y = 'saved_datas/y_tot.csv'
    out_x.to_csv(os.path.join(path_x),index = False)
    out_y.to_csv(os.path.join(path_y),index = False)

#when p_datastate is set to 'load'
def load_xy():
    print('Loading the X and y...')
    X_tot = (pd.read_csv('saved_datas/X_tot.csv')).to_numpy()
    y_tot = (pd.read_csv('saved_datas/y_tot.csv')).to_numpy()
    return X_tot,y_tot

########################################STARTOFCODE##########################
print('##############################START##############################')
data_lab = pd.read_csv("input_data/csv/train_labeled.csv")

data_unlab = pd.read_csv("input_data/csv/train_unlabeled.csv")

X_sub_pd = pd.read_csv("input_data/csv/test.csv")

#Tensorboard/log part
logs_base_dir = "./logs"
os.makedirs(logs_base_dir,exist_ok=True)
#Build 

output_name = build_output_name()

log_spec = os.path.join(logs_base_dir,output_name)
os.makedirs(log_spec,exist_ok=True)

###############################SEMI SUPERVISED PART############################
if (p_datastate == 'save'):
  RESULT_ACC_SS = 0
  for i in range (p_manyfit):
    init_variables()
    #PCA preprocessing
    if (PCA_MODE):
        pca_preprocess()
    #Semi supervised algo
    if (p_ss_mod=='LabSpr' and p_ss_kern=='knn'):
            label_prop_model = dic_ss_mod[p_ss_mod](kernel=p_ss_kern,gamma=p_gamma,n_neighbors=p_neighbors,alpha=p_alpha)
    elif (p_ss_mod=='LabSpr' and p_ss_kern=='rbf'):
            label_prop_model = dic_ss_mod[p_ss_mod](kernel=p_ss_kern,gamma=p_gamma,n_neighbors=p_neighbors,alpha=p_alpha,max_iter=1)
    else:            
        label_prop_model = dic_ss_mod[p_ss_mod](kernel=p_ss_kern,gamma=p_gamma,n_neighbors=p_neighbors)
    print('Start to fit. Run for shelter!')
    label_prop_model.fit(X_tot,y_tot)
    temp_acc = label_prop_model.score(X_valid_lab,y_valid)
    print('{} / {} :accuracy = {}'.format(i,p_manyfit,temp_acc))
    RESULT_ACC_SS += temp_acc  
  y_tot = label_prop_model.transduction_
  y_submit = label_prop_model.predict(X_submit)
  save_to_csv(X_tot,y_tot)
  RESULT_ACC_SS /= p_manyfit
  json_dict['ss_accuracy'] = RESULT_ACC_SS
  print('accuracy obtained on the test set of the ss algo:',RESULT_ACC_SS)
else:
  init_variables()
  #PCA preprocessing
  if (PCA_MODE):
      pca_preprocess()
  X_tot,y_tot = load_xy()

##############################NEURAL NETWORK PART ##################################

if (USING_NN):
    model = build_model()

    call_back_list = [] 
    #Euler doesn't like the modelcheckpoint
    #call_back_list.append(ModelCheckpoint(log_spec+'/best_mod.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True))

    call_back_list.append(keras.callbacks.TensorBoard(log_spec,histogram_freq=1,write_grads=True))

    if (EARLY_STOP_MODE):
        call_back_list.append(EarlyStopping( patience=p_patience, verbose=1, mode='min'))

    model.fit(x=X_tot,
            y=y_tot,
            epochs = p_epochs,
            batch_size=p_batch_size,
            validation_data=(X_valid_lab,y_valid),
            callbacks=call_back_list)
    #model.fit(X_train, y_train,epochs = p_epochs, shuffle=False ,validation_split = 0.1)

    test_loss, aut_acc = model.evaluate(X_valid_lab,y_valid)

    y_probas = (model.predict(X_valid_lab))

    y_valid_predicted = np.array([np.argmax(i) for i in y_probas])

    RESULT_ACC_NN = accuracy_score(y_valid_predicted, y_valid)

    json_dict['nn_accuracy'] = RESULT_ACC_NN

    #predict the output
    y_submit_conf = model.predict(X_submit) #for each input, we still have only the confidences, we need to take the max one
    y_submit = np.array([np.argmax(i) for i in y_submit_conf])

###########################OUTPUT:########################################
#creates the output name
submission_formed(y_submit,output_name)

with open(log_spec+'/recap.json','w') as fp:
    json.dump(json_dict, fp, indent = 1)
print('########################################DONE##################################')
print("\n")
