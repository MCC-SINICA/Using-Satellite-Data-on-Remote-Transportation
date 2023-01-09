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

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



start_time = time.time()

img_shape = (30,38)
n_channels = 1
n_classes = 18
#optim = Adam(lr = 0.0015)
optim = Adam(lr = 0.0001, decay = 0.0001)
#sample_batch = 16
sample_batch = 10
time_step = 4
number_stations = 18
NB_epoch = 500

def eval_metrics_on(predictions,label_localstations_pm25s):
    '''assuming this is a regression task; label_localstations_pm25s are continuous-valued floats
    
    returns most regression-related scores for the given predictions/targets as a dictionary:
    
        r2, mean_abs_error, mse, rmse, median_absolute_error, explained_variance_score
    '''
    #if len(label_localstations_pm25s[0])==2: #label_localstations_pm25s is list of data/label_localstations_pm25s pairs
        #label_localstations_pm25s = np.concatenate([l[1] for l in label_localstations_pm25s])
    #predictions = predictions[:,0]
    prdictions = np.array(predictions)
    label_localstations_pm25s = np.array(label_localstations_pm25s)
    
    r2                       = metrics.r2_score(label_localstations_pm25s, predictions)
    mean_abs_error           = np.abs(predictions - label_localstations_pm25s).mean()
    mse                      = ((predictions - label_localstations_pm25s)**2).mean()
    rmse                     = np.sqrt(mse)
    #median_absolute_error    = metrics.median_absolute_error(label_localstations_pm25s, predictions) # robust to outliers
    explained_variance_score = metrics.explained_variance_score(label_localstations_pm25s, predictions) # best score = 1, lower is worse
    return {'r2':r2, 'mean_abs_error':mean_abs_error, 'mse':mse, 'rmse':rmse, 
    'explained_variance_score':explained_variance_score}


def predict_graph(true_val,predict_val):
    for sn in range(18):
        plt.figure(figsize=(6,3))
        plt.figure(sn+1)
        plt.title('_next_4hour station = ' + str(sn+1))
        plt.plot(true_val[:,sn::18], c='g', label_localstations_pm25= 'gt')
        plt.plot(predict_val[:,sn::18], c='r', label_localstations_pm25= 'pred')
        plt.legend(loc='best',prop={'size': 5})
        plt.legend(loc='best')
        name = str("firstgroup_stat_") + str(sn) + str(".png")
        #plt.savefig(name)
        plt.show()
        
aod_train1 = np.load('middlelayer_foryear14_4hr_4tile.npy')
aod_train2 = np.load('middlelayer_foryear15_4hr_4tile.npy')
aod_train = np.concatenate((aod_train1,aod_train2), axis = 0)
aod_valid = np.load('middlelayer_foryear16_4hr_4tile.npy')
pm25_train = np.load('pm_train_4hr_.npy')
pm25_train = pm25_train.reshape(pm25_train.shape[0]*pm25_train.shape[1],pm25_train.shape[2])
pm25_valid = np.load('pm_valid_4hr.npy')
pm25_valid = pm25_valid.reshape(pm25_valid.shape[0]*pm25_valid.shape[1],pm25_valid.shape[2])
weath_train = np.load('localweather_train_4hr.npy')
future_weath_train = np.load('future_localweather_train_4hr.npy')
weath_valid = np.load('localweather_test_4hr.npy')
future_weath_valid = np.load('future_localweather_test_4hr.npy')
label_localstations_pm25_train = np.load('localstations_pm25_4hr.npy')
label_localstations_pm25_valid = np.load('label_localstations_pm25_valid_4hr.npy')

model.compile(optimizer = optim, loss = 'mse', metrics = ['mse'])

keras_callback = [
                EarlyStopping(monitor='val_mse', patience = 90, mode='min'),
                ModelCheckpoint(filepath='next4hr_stri_p.h5', monitor='val_mse', 
                save_best_only=True,mode='min',verbose=1)
                ]



history = model.fit([aod_train.astype('float'),pm25_train.astype('float'),weath_train.astype('float'),
                     future_weath_train.astype('float')],
                            label_localstations_pm25_train.astype('float'),batch_size = sample_batch,
                            epochs = NB_epoch,verbose = 1,shuffle = False,callbacks=keras_callback,
            validation_data = ([aod_valid.astype('float'),pm25_valid.astype('float'),weath_valid.astype('float'),
                                future_weath_valid.astype('float')],
                                label_localstations_pm25_valid.astype('float')))


#save model train history
np.save('history_n4hr_stri.npy',history.history)                         
                            

#save model
file_name = 'next4hr_stri_p_model.h5'
model.save(file_name)


pred = model.predict([aod_valid.astype('float'),pm25_valid.astype('float'),weath_valid.astype('float'),
                      future_weath_valid.astype('float')],batch_size = sample_batch, verbose=1)


predictions = pred.reshape(pred.shape[0]*pred.shape[1],pred.shape[2])

y_label_localstations_pm25 = label_localstations_pm25_valid
y_label_localstations_pm252 = y_label_localstations_pm25.reshape(y_label_localstations_pm25.shape[0]*y_label_localstations_pm25.shape[1],y_label_localstations_pm25.shape[2])
print('shape of true and predict ',y_label_localstations_pm252.shape,predictions.shape)

eval_results = eval_metrics_on(predictions,y_label_localstations_pm25)
print('this is evaluation results for r2,mean_abs_error,mse,rmse,expl_var_score ',eval_results)

#observe the graph between prediction and true value of pm25 for last hour
y_label_localstations_pm25 = np.load('label_localstations_pm25_valid_4hr.npy')
y_true = y_label_localstations_pm25[:,3:4,:]
y_true = y_true.reshape(y_true.shape[0]*y_true.shape[1],y_true.shape[2])
pred_graph = predictions_edit[:,3:4,:]
pred_graph = pred_graph.reshape(pred_graph.shape[0]*pred_graph.shape[1],pred_graph.shape[2])
predict_graph(y_true,pred_graph)


print('rmse value for each time step from next1hr to 4hr ')
stat_dict = {}
stat_avg = []
for i in range(time_step2):
    t1 = predictions_edit[:,i:i+1,:]
    t1 = t1.reshape(t1.shape[0]*t1.shape[1],t1.shape[2])
    t2 = y_label_localstations_pm25[:,i:i+1,:]
    t2 = t2.reshape(t2.shape[0]*t2.shape[1],t2.shape[2])
    rsl = []
    for sn in range(2):
        name = 'stats_' + str(sn)
        score2 = np.sqrt(np.mean(np.square(t1[:,sn::2]-t2[:,sn::2])))
        stat_dict.setdefault(name,[]).append(score2)
        #print(score2)
        rsl.append(score2)
for k in  stat_dict.keys():
    avg_value = sum(stat_dict[k])/time_step2
    #print(avg_value)
    stat_avg.append(avg_value)
print('mean of all stations ',mean(stat_avg))

#save prediction results to be used in composite model
np.save('stri_p_4hr_prediction_results_yr16.npy',pred)

                    #END

#########################################################################################
#CODE BELOW IS FOR CALCULATING INPUT IN COMPOSITE MODEL USING STRI_P SAVED Model

#RUN SEPARATE FOR USING INPUT FOR YEAR 2014 and 2015
#########################################################################################


from tensorflow.keras import backend as K
from tensorflow import keras

import numpy as np
from tensorflow.keras import losses

from tensorflow.keras.utils import plot_model
import tensorflow.python.keras.backend as K

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import *
K.set_image_data_format('channels_first')

import seaborn as sns
sns.set()

import os

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



sample_batch = 10

#input feature for year 2014             
aod_train_year14 = np.load('middlelayer_foryear14_4hr_4tile.npy')
pm25_train_yr14 = np.load('pm_train_yr14.npy')
pm25_train_yr14 = pm25_train_yr14.reshape(pm25_train_yr14.shape[0]*pm25_train_yr14.shape[1],pm25_train_yr14.shape[2])
label_localstations_pm25_train_yr14 = np.load('localstations_pm25_yr14.npy')
weath_yr14 = np.load('localweather_yr14_.npy')
future_weath_yr14 = np.load('future_localweather_yr14_.npy')

#import the stri_p model
model2 = load_model('next4hr_stri_p_model.h5')
pred = model2.predict([aod_train_yr14.astype('float'),pm25_train_yr14.astype('float'),weath_yr14.astype('float'),
    future_weath_yr14.astype('float')],batch_size = sample_batch, verbose=1)

#save the prediction results as input to composite model
np.save('stri_p_4hr_prediction_results_yr14.npy',pred)


######################################################################################################################

from tensorflow.keras import backend as K
from tensorflow import keras

import numpy as np
from tensorflow.keras import losses

from tensorflow.keras.utils import plot_model
import tensorflow.python.keras.backend as K

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import *
K.set_image_data_format('channels_first')

import seaborn as sns
sns.set()

import os

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



sample_batch = 10

#input feature for year 2014             
aod_train_year15 = np.load('middlelayer_foryear15_4hr_4tile.npy')
pm25_train_yr15 = np.load('pm_train_yr15.npy')
pm25_train_yr15 = pm25_train_yr15.reshape(pm25_train_yr15.shape[0]*pm25_train_yr15.shape[1],pm25_train_yr15.shape[2])
label_localstations_pm25_train_yr15 = np.load('localstations_pm25_yr15.npy')
weath_yr15 = np.load('localweather_yr15_.npy')
future_weath_yr15 = np.load('future_localweather_yr15_.npy')

#import the stri_p model
model3 = load_model('next4hr_stri_p_model.h5')
pred = model3.predict([aod_train_yr15.astype('float'),pm25_train_yr15.astype('float'),weath_yr15.astype('float'),
    future_weath_yr15.astype('float')],batch_size = sample_batch, verbose=1)

#save the prediction results as input to composite model
np.save('stri_p_4hr_prediction_results_yr15.npy',pred)
