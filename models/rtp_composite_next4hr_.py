from tensorflow.keras import backend as K
from tensorflow import keras

import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib import pyplot

import threading
import argparse
import pickle
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn.metrics as metrics
import math
import time
import gc
from datetime import datetime
from collections import Iterable
from tensorflow.keras import losses
from statistics import mean
from tensorflow.keras.utils import plot_model
import tensorflow.python.keras.backend as K

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint, EarlyStopping,Callback
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import *
K.set_image_data_format('channels_first')

import seaborn as sns
sns.set()

import os


import tensorflow.compat.v1 as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

img_shape = (30,38)
n_channels = 1
n_classes = 18
optim = Adam(lr = 0.001, decay = 0.0001)
#sample_batch = 16
sample_batch = 40
time_step = 4
number_stations = 18
NB_epoch = 800



def eval_metrics_on(predictions,pm25_labels):
    '''assuming this is a regression task; pm25_labels are continuous-valued floats
    
    returns most regression-related scores for the given predictions/targets as a dictionary:
    
        r2, mean_abs_error, mse, rmse, median_absolute_error, explained_variance_score
    '''
    #if len(pm25_labels[0])==2: #pm25_labels is list of data/pm25_labels pairs
        #pm25_labels = np.concatenate([l[1] for l in pm25_labels])
    #predictions = predictions[:,0]
    prdictions = np.array(predictions)
    pm25_labels = np.array(pm25_labels)
    
    r2                       = metrics.r2_score(pm25_labels, predictions)
    mean_abs_error           = np.abs(predictions - pm25_labels).mean()
    mse                      = ((predictions - pm25_labels)**2).mean()
    rmse                     = np.sqrt(mse)
    #median_absolute_error    = metrics.median_absolute_error(pm25_labels, predictions) # robust to outliers
    explained_variance_score = metrics.explained_variance_score(pm25_labels, predictions) # best score = 1, lower is worse
    return {'r2':r2, 'mean_abs_error':mean_abs_error, 'mse':mse, 'rmse':rmse, 
    'explained_variance_score':explained_variance_score}


def predict_graph(true_val,predict_val):
    for sn in range(18):
        plt.figure(figsize=(6,3))
        plt.figure(sn+1)
        plt.title('_next_16hour station = ' + str(sn+1))
        plt.plot(true_val[:,sn::18], c='g', pm25_label= 'gt')
        plt.plot(predict_val[:,sn::18], c='r', pm25_label= 'pred')
        plt.legend(loc='best',prop={'size': 5})
        plt.legend(loc='best')
        name = str("onemany_stat_") + str(sn) + str(".png")
        #plt.savefig(name)
        plt.show()
        

        
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()
        
data_shape = (4,18)
in1 = Input(shape=data_shape)

# Model 2
data_shape1 = (4,18)
in2 = Input(shape=data_shape1)

model_final_concat = concatenate([in1,in2])


model_final_concat = TimeDistributed(Dense(54, activation='relu',bias_regularizer=L1L2(l1 = 0.01,l2=0.01)))(model_final_concat)
model_final_concat = TimeDistributed(Dense(40, activation='relu',bias_regularizer=L1L2(l1 = 0.01,l2=0.01)))(model_final_concat)
model_final_concat = Dropout(rate=0.2)(model_final_concat)

model_final_concat = TimeDistributed(Dense(n_classes))(model_final_concat)


model = Model(inputs=[in1,in2],outputs=model_final_concat)



model.summary()




strip_train1 = np.load('stri_p_4hr_prediction_results_yr14.npy')
strip_train2 = np.load('stri_p_4hr_prediction_results_yr15.npy')
strip_train = np.concatenate((strip_train1,strip_train2), axis = 0)
strip_valid = np.load('stri_p_4hr_prediction_results_yr16.npy')
basemodel_train1 = np.load('basemodel_4hr_prediction_yr14_.npy')
basemodel_train2 = np.load('basemodel_4hr_prediction_yr15_.npy')
basemodel_train = np.concatenate((basemodel_train1,basemodel_train2), axis = 0)
basemodel_valid = np.load('basemodel_4hr_prediction_yr16_.npy')
basemodel_valid = basemodel_valid[:,0:4,:]
pm25_label_train1 = np.load('pm25_labe14_4hr.npy')
pm25_label_train2 = np.load('pm25_label15_4hr.npy')
pm25_label_train = np.concatenate((pm25_label_train1,pm25_label_train2), axis = 0)
pm25_label_valid = np.load('pm25_label16_4hr.npy')


model.compile(optimizer = optim, loss = 'mse', metrics = ['mse'])

keras_callback = [
                EarlyStopping(monitor='val_mse', patience = 50, mode='min'),
                ModelCheckpoint(filepath='next4hr_composite_.h5', monitor='val_mse', 
                save_best_only=True,mode='min',verbose=1)
                ]


history = model.fit([strip_train.astype('float'),basemodel_train.astype('float')],
                            pm25_label_train.astype('float'),batch_size = sample_batch,
                            epochs = NB_epoch,verbose = 1,shuffle = False,callbacks=keras_callback,
            validation_data = ([strip_valid.astype('float'),basemodel_valid.astype('float')],
                                pm25_label_valid.astype('float')))


#save model train history
np.save('history_n4hr_composite.npy',history.history)                         
                            

#save model
file_name = 'next4hr_composite_model.h5'
model.save(file_name)



pred = model.predict([strip_valid.astype('float'),basemodel_valid.astype('float')],batch_size = sample_batch, verbose=1)

predictions = pred.reshape(pred.shape[0]*pred.shape[1],pred.shape[2])

print(pred.shape)
y_pm25_label = pm25_label_valid
y_pm25_label2 = y_pm25_label.reshape(y_pm25_label.shape[0]*y_pm25_label.shape[1],y_pm25_label.shape[2])

print('shape of true and predict ',y_pm25_label2.shape,predictions.shape)
eval_results = eval_metrics_on(predictions,y_pm25_label2)
print('this is evaluation results for r2,mean_abs_error,mse,rmse,expl_var_score ',eval_results)

#observe the graph between prediction and true value of PM2.5 for last hour
y_true = y_pm25_label[:,3:4,:]
y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1],y_true.shape[2])
y_true = y_true
pred_graph = pred[:,3:4,:]
pred_graph =pred_graph.reshape(pred_graph.shape[0]*pred_graph.shape[1],pred_graph.shape[2])
predict_graph(y_true,pred_graph)


print('rmse value for each time step from next1hr to 4hr ')
stat_dict = {}
stat_avg = []

for i in range(time_step):
    t1 = pred[:,i:i+1,:]
    t1 = t1.reshape(t1.shape[0]*t1.shape[1],t1.shape[2])
    t2 = y_pm25_label[:,i:i+1,:]
    t2 = t2.reshape(t2.shape[0]*t2.shape[1],t2.shape[2])
    rsl = []
    for sn in range(18):
        name = 'stats_' + str(sn)
        score2 = np.sqrt(np.mean(np.square(t1[:,sn::18]-t2[:,sn::18])))
        stat_dict.setdefault(name,[]).append(score2)
        #print(score2)
        rsl.append(score2)
for k in  stat_dict.keys():
    avg_value = sum(stat_dict[k])/time_step
    #print(avg_value)
    stat_avg.append(avg_value)
print('mean of all stations ',mean(stat_avg))