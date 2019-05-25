import random
import json
import numpy as np
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
import keras
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Dense, Activation
from keras.optimizers import SGD, Adam
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
#tf.set_random_seed(14)
#np.random.seed(seed=14)
#random.seed(14)

class SemiSupLabeler():

    def __init__(self,data_lab,data_unlab,data_submit):
      ###########################Default parameters#####################
      #NB:if some mandatory parameters are lacking in the json, default values will be taken
      #list of all potential parameters
      self.params_nn = ['loss','optimizer','learning rate','metrics','decay','momentum','batch_size','number of epochs','layers','patience']
      #loss:              loss used for the NN, cf the dictionnary above
      #optimizer:         Adam,SGD etc, cf the dictionnary above
      #learning rate  
      #metrics            accuracy, we wont change it normally
      #decay:             decay of the learning rate, generally of the order 1e-5
      #momentum:          momentum of the lr
      #patience:          number of epochs you wait if you use earlystopmode for the validation accuracy to increase again
      #layers:            shape of the network

      self.params_ss = ['UsingSS','manyfit','ss_model','ss_kernel','gamma','neighbor','alpha']
      #manyfit:           since the ss accuracy has some variance but doesnt take much to be computed, manyfit designs how many independant times we run it before averaging it in order to obtain a better estimation of the accuracy in question
      #ss_model:          'LabSpr' or 'LabProp'. So far, only LabSpr has converged
      #ss_kernel:         'knn' or 'rbf. So far only knn converges. ***WATCH OUT***: when using rbf, euler will complain that you use too much memory!!
      #gamma              parameter for the rbf
      #neighbor           parameter for knn
      #alpha              parameter for knn and rbf: tells at which point you will take the information of your neighbors into account

      self.param_list = ['Ratio','pca','UsingNN','paramsout','data_state','scaler']+self.params_nn+self.params_ss
      #Ratio              ratio represented by the training set
      #pca                number of principal components to use. if not present, no pca will be done
      #UsingNN            if set to false, the NN is not used.
      #data_state         'save' or 'load'. If you want to train the NN only without having to run the ss algo again, do one run with data_state to true, and use data_state= 'load for the next ones.
      #scaler             'normal' or 'standard' describes the preprocessing before applying the pca
      #paramsout:         designates which parameters will be present in the output name ==> put the one you're playing with in order to easily see the difference
      self.param_out = ['Ratio','pca','optimizer','layers']

      self.data_lab = data_lab
      self.data_unlab = data_unlab
      self.data_submit = data_submit
      #(1)Extracting the data
      #Training
      self.RATIO = 0.9
      self.INPUT_DIM = 139

      #PCA
      self.scaler = 'Standard'
      self.PCA_MODE = True
      self.pca = 50

      #Early stopping
      self.EARLY_STOP_MODE = False
      self.patience = 50

      #NN
      self.USING_NN = True
      self.USING_SS = False
      assert(self.USING_NN or self.USING_SS)
      self.loss = "sparse_categorical_crossentropy"
      self.opt = "SGD"
      self.lr = 0.001
      self.metric = "accuracy"
      self.decay = 0
      self.momentum = 0
      self.batch_size = 32
      self.epochs = 5
      self.lay_node = [("relu",206),('dropout',0.33)]

      #Semi Supervised 
      self.datastate = 'save'
      self.ss_mod = 'LabSpr'
      self.ss_kern = 'knn'
      self.gamma = 20
      self.neighbors = 7
      self.alpha = 0.2
      self.manyfit = 1
      ###################################################################

      def check(inner,outer):
          for i in inner:
              if not (i in outer): 
                  print('unknown parameter. abort.',i)
                  exit()

      self.JSON_MODE = (len(sys.argv) >1)
      if (self.JSON_MODE):
          fn = sys.argv[1]
          if os.path.isfile(fn):
              print("successfully read the json file."+sys.argv[1])
              self.json_dict = json.load(open(fn))
              assert ('UsingNN' and 'paramsout' in self.json_dict)
              self.USING_NN = self.json_dict['UsingNN']
              self.USING_SS = self.json_dict['UsingSS']
              check(self.json_dict,self.param_list)
              check(self.json_dict['paramsout'],self.param_list)
              #iterate over the printed parameters and ensure they exist
              self.param_out = self.json_dict['paramsout']
              self.RATIO = self.json_dict['Ratio']
              self.ss_mod = self.json_dict['ss_model']
              self.ss_kern = self.json_dict['ss_kernel']
              self.gamma = self.json_dict['gamma']
              self.neighbors = self.json_dict['neighbor']
              self.alpha = self.json_dict['alpha']
              self.datastate = self.json_dict['data_state']
              self.scaler = self.json_dict['scaler']
              if ('manyfit' in self.json_dict):
                self.manyfit = self.json_dict['manyfit']
              if (self.USING_NN):
                  self.loss = self.json_dict['loss']
                  self.opt = self.json_dict['optimizer']
                  self.lr = self.json_dict['learning rate']
                  self.metric = self.json_dict['metrics']
                  self.decay = self.json_dict['decay']
                  self.momentum = self.json_dict['momentum']
                  self.batch_size = self.json_dict['batch_size']
                  self.epochs = self.json_dict['number of epochs']
                  lay_node = self.json_dict['layers']
              self.PCA_MODE = ('pca' in self.json_dict)
              if (self.PCA_MODE):
                  self.pca = self.json_dict['pca']
                  self.INPUT_DIM = self.pca
              self.EARLY_STOP_MODE = ('patience' in self.json_dict)
              if (self.EARLY_STOP_MODE):
                  self.patience = self.json_dict['patience']
          else:
              print("uncorrect path. abort.")
              print(sys.argv[1])
              exit()
      else :
          print("taking the values of the code")
          self.json_dict = {
                  'Ratio':self.RATIO,
                  'UsingNN': self.USING_NN,
                  'UsingSS': self.USING_SS,
                  'ss_model':self.ss_mod,
                  'ss_kernel':self.ss_kern,
                  'loss':self.loss,
                  'optimizer':self.opt,
                  'learning rate' :self.lr,
                  'metrics':self.metric,
                  'decay':self.decay,
                  'momentum':self.momentum,
                  'batch_size':self.batch_size,
                  'number of epochs': self.epochs,
                  'gamma':self.gamma,
                  'neighbor':self.neighbors,
                  'alpha':self.alpha,
                  'layers':self.lay_node,
                  'manyfit':self.manyfit,
                  'scaler':self.scaler
                  }
          if (self.PCA_MODE):
              self.json_dict['pca'] = self.pca
              self.INPUT_DIM = self.pca
          if (self.EARLY_STOP_MODE):
              self.jsondict['patience'] = self.patience
      self.build_output_name()
      #Tensorboard/log part
      self.logs_base_dir = "./logs"
      os.makedirs(self.logs_base_dir,exist_ok=True)
      self.log_spec = os.path.join(self.logs_base_dir,self.output_name)
      os.makedirs(self.log_spec,exist_ok=True)
      self.init_variables()



    def label_spr(self):
      RESULT_ACC_SS = 0
      for i in range (self.manyfit):
        self.init_variables()
        #PCA preprocessing
        if (self.PCA_MODE):
            self.pca_preprocess(self.pca)
        #Semi supervised algo
        if (self.ss_mod=='LabSpr' and self.ss_kern=='knn'):
                self.label_prop_model = LabelSpreading(kernel='knn',gamma=self.gamma,n_neighbors=self.neighbors,alpha=self.alpha)
        elif (self.ss_mod=='LabProp' and self.ss_kern=='rbf'):
                self.label_prop_model = LabelPropagation(kernel='rbf',gamma=self.gamma,n_neighbors=self.neighbors,alpha=self.alpha,max_iter=10)
        else:            
            self.label_prop_model = LabelPropagtion(kernel=self.ss_kern,gamma=self.gamma,n_neighbors=self.neighbors)
        print('Start to fit. Run for shelter!')
        self.label_prop_model.fit(self.X_tot,self.y_tot)
        temp_acc = self.label_prop_model.score(self.X_valid_lab,self.y_valid)
        print('{} / {} :accuracy = {}'.format(i,self.manyfit,temp_acc))
        RESULT_ACC_SS += temp_acc   

      self.y_tot = label_prop_model.transduction_
      self.y_submit = label_prop_model.predict(X_submit)
      if (self.data_state == "save"):
        self.save_to_csv(self.X_tot,self.y_tot)
      RESULT_ACC_SS /= self.manyfit
      self.json_dict['ss_accuracy'] = RESULT_ACC_SS
      print('accuracy obtained on the test set of the ss algo:',RESULT_ACC_SS)

    def labelspr_predict(self,X):
      return self.label_prop_model.predict(X)

    def init_variables(self):
      X_submit = self.data_submit.to_numpy()
      X_big_lab = (self.data_lab.to_numpy())[:,1:]
      y_big = ((self.data_lab.to_numpy())[:,0]).astype(int)
      X_train_lab, X_valid_lab,self.y_train,self.y_valid = train_test_split(X_big_lab,y_big,test_size = (1-self.RATIO))#,random_state=14)
      X_unlab = self.data_unlab.to_numpy()
      X_tot = np.concatenate((X_train_lab,X_unlab),axis=0)
      self.y_tot = np.concatenate((self.y_train,np.full(len(X_unlab),-1)))
      if (self.scaler == 'Standard'):
         scaler = StandardScaler()
      elif (self.scaler == 'Normal'):
         scaler = Normalizer()
      else:
         scaler = StandardScaler()
      self.X_tot = scaler.fit_transform(X_tot)
      self.X_train_lab = scaler.transform(X_train_lab)
      self.X_unlab = scaler.transform(X_unlab)
      self.X_valid_lab = scaler.transform(X_valid_lab)
      self.X_submit = scaler.transform(X_submit)

    def pca_preprocess(self,number): 
        pca_mod = PCA(n_components= number)
        self.X_tot = pca_mod.fit_transform(self.X_tot)
        self.X_train_lab = pca_mod.transform(self.X_train_lab)
        self.X_unlab = pca_mod.transform(self.X_unlab)
        self.X_valid_lab = pca_mod.transform(self.X_valid_lab)
        self.X_submit = pca_mod.transform(self.X_submit)
        self.INPUT_DIM = number

    def build_model(self):
        self.model = Sequential()
        for counter,(name,num) in enumerate(self.lay_node):
            if (counter==0):
                self.model.add(Dense(num,activation='relu',input_dim=self.INPUT_DIM))
            elif (name == 'dropout'):
                self.model.add(Dropout(rate=num))
            elif (name=='relu'):
                self.model.add(Dense(num,activation=tf.nn.relu))
            elif (name=='relu_bn'):
                self.model.add(Dense(num))
                self.model.add(BatchNormalization()) 
                self.model.add(Activation('relu'))
            else:
                print('uncorrect name for the layers. exit.')
                exit()
        self.model.add(Dense(10,activation='softmax'))
        #optimizer
        if (self.opt=='SGD'):
            optimiz = SGD(lr = self.lr,decay = self.decay,momentum = self.momentum)
        elif (self.opt == 'Adam') :
            optimiz = Adam (lr = self.lr,decay = self.decay)
        else :
            print('uncorrect name for the layers. exit.')
            exit()
        self.model.compile(optimizer = optimiz,
                     loss = self.loss,
                     metrics=[self.metric])
        
    def fit_lab(self):
      temp = self.nn_fit(self.X_train_lab,self.y_train)
      self.json_dict["small_lab_dataset_nn_acc"] = temp

    def fit_tot(self):
      temp = self.nn_fit(self.X_tot,self.y_tot)
      self.json_dict["big_dataset_nn_acc"] = temp

    def nn_fit(self,X,y):
      call_back_list = [] 
      #call_back_list.append(keras.callbacks.TensorBoard(self.log_spec,histogram_freq=1,write_grads=True))
      if (self.EARLY_STOP_MODE):
          call_back_list.append(EarlyStopping( patience=self.patience, verbose=1, mode='min',restore_best_weights=True))

      self.model.fit(x=X,
              y=y,
              epochs = self.epochs,
              batch_size=self.batch_size,
              validation_data=(self.X_valid_lab,self.y_valid))
      test_loss, aut_acc = self.model.evaluate(self.X_valid_lab,self.y_valid)
      y_temp = self.model.predict(self.X_submit)
      self.y_submit = np.array([np.argmax(i) for i in y_temp]) 
      return aut_acc
      


    def complete_unlab(self):
      y_missing = self.model.predict(self.X_unlab)
      y_missing = np.array([np.argmax(i) for i in y_missing])
      self.X_tot = np.concatenate((self.X_train_lab,self.X_unlab),axis=0)
      self.y_tot = np.concatenate((self.y_train,y_missing),axis=0)


    def build_output_name(self):
        self.output_name = (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if (self.JSON_MODE):
            if ('origin' in os.path.basename(os.path.normpath(sys.argv[1]))):
                self.output_name+='_OR_'
            nn_string = 'NN:'
            ss_string = 'SS:'
            for i in self.param_out:
                temp = (i + '=' + str(self.json_dict[i]))
                if ( i in self.params_nn):
                    nn_string += temp     
                elif ( i in self.params_ss):
                    ss_string += temp 
                else:
                    self.output_name += temp
            self.output_name += ss_string
            if (self.USING_NN):
                self.output_name += nn_string

    def submission_formed(self,predicted_y,name ):
        result_dir = "./results"
        os.makedirs(result_dir,exist_ok=True)
        out = pd.DataFrame(predicted_y)
        out.insert(0,'Id',range(30000,len(out)+30000))
        out.rename(columns={"Id": "Id", 0: "y"},inplace = True)
        path = 'results/'+name+'.csv'
        out.to_csv(os.path.join(path),index = False)

    #useful when self.datastate is set to 'save': save the datas obtained after the ss algoh

    def save_to_csv(self,X_tot,y_tot,X_valid,y_valid):
        out_x = pd.DataFrame(X_tot)
        out_y = pd.DataFrame(y_tot) 
        out_xv = pd.DataFrame(X_valid)
        out_yv = pd.DataFrame(y_valid) 
        os.makedirs('./saved_datas',exist_ok=True)
        path_x = 'saved_datas/X_tot.csv'
        path_y = 'saved_datas/y_tot.csv'
        path_xv = 'saved_datas/X_valid.csv'
        path_yv = 'saved_datas/y_valid.csv'
        out_x.to_csv(os.path.join(path_x),index = False)
        out_y.to_csv(os.path.join(path_y),index = False)
        out_xv.to_csv(os.path.join(path_xv),index = False)
        out_yv.to_csv(os.path.join(path_yv),index = False)

    #when self.datastate is set to 'load'
    def load_xy(self):
        print('Loading the X and y...')
        self.X_valid_lab =(pd.read_csv('saved_datas/X_valid.csv')).to_numpy()
        self.y_valid = (pd.read_csv('saved_datas/y_valid.csv')).to_numpy()
        self.X_tot = (pd.read_csv('saved_datas/X_tot.csv')).to_numpy()
        self.y_tot = (pd.read_csv('saved_datas/y_tot.csv')).to_numpy()

    def out(self):
      self.submission_formed(self.y_submit,self.output_name)

      with open(self.log_spec+'/recap.json','w') as fp:
          json.dump(self.json_dict, fp, indent = 1)
      print('########################################DONE##################################')
      print("\n")

########################################STARTOFCODE##########################
print('##############################START##############################')
data_lab = pd.read_csv("input_data/csv/train_labeled.csv")

data_unlab = pd.read_csv("input_data/csv/train_unlabeled.csv")

data_submit = pd.read_csv("input_data/csv/test.csv")

machine = SemiSupLabeler(data_lab,data_unlab,data_submit)

#Build 
if (machine.datastate == "load"):
  machine.load_xy()
if (machine.PCA_MODE):
  print('DOING PCA')
  machine.pca_preprocess(machine.pca)
if (machine.USING_SS):
   machine.label_spr()
if (machine.USING_NN):
  machine.build_model()
  machine.fit_lab()
  machine.complete_unlab()
  machine.fit_tot()
machine.out()

