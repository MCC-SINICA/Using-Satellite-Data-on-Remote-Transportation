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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



start_time = time.time()

#img_shape = (1200,1200)
n_channels = 1
n_classes = 18
#optim = Adam(lr = 0.0015)
sample_batch = 10
time_step = 1
time_step2 = 4
number_stations = 18
optim = Adam(lr = 0.00001, decay = 0.00001)
# Setup the model



            
def pred_generat(tile28v05,tile28v06,tile29v05,tile29v06,tile31v06,pm,batch_size,num_channels,
                  num_channels_2,num_channels_3,
                  time,time_label,cover_hr_in=1):
    
    while True:
        #x_shape_1 = (batch_size,num_channels,time_label,600,1200)
        x_shape_1 = (batch_size,time_label,num_channels,300,300)
        
        #x_shape_2 = (batch_size,num_channels,time_label,650,600)
        x_shape_2 = (batch_size,time_label,num_channels,300,300)
        
        x_shape_3 = (batch_size,time,num_channels_2,2,7)
        
        x_shape_4 = (batch_size,time_label,num_channels_3,4,6)
        
        #y_shape = (batch_size,time,18)
        y_shape = (batch_size,time,37)
        
        
        h28v05 = np.zeros(shape=x_shape_1)
        h28v06 = np.zeros(shape=x_shape_1)
        
        h29v05 = np.zeros(shape=x_shape_2)
        h29v06 = np.zeros(shape=x_shape_2)
        
        h31v06 = np.zeros(shape=x_shape_3)
        #h32v06 = np.zeros(shape=x_shape_4)
        pm_batch = np.zeros(shape=y_shape)
        
        
        #pm = np.zeros(shape=y_shape)
        y_batch = np.zeros(shape=y_shape)
        
        end = pm.shape[0]
        #print('this is end ', end)
        
        
        for batch_idx in range(0,end,int(batch_size)):
            for seq_idx in range(int(batch_size)):
                idx = batch_idx + seq_idx
                
                h28v05[seq_idx] = tile28v05[idx:idx+cover_hr_in][0]
                h28v06[seq_idx] = tile28v06[idx:idx+cover_hr_in][0]
                
                h29v05[seq_idx] = tile29v05[idx:idx+cover_hr_in][0]
                h29v06[seq_idx] = tile29v06[idx:idx+cover_hr_in][0]
                
                h31v06[seq_idx] = tile31v06[idx:idx+cover_hr_in][0]
                pm_batch[seq_idx] = pm[idx:idx+cover_hr_in][0]

            
            yield ([h28v05,h28v06,h29v05,h29v06,h31v06,pm_batch])
            
            


chan = 1
chan2 = 148
chan3 = 2
time_t = 4

#load saved model
model = load_model('next4hr_4tiles_18stations.h5')

#feature dataset for year 2014
h28v5_16 = (np.load('h28v06_16_train.npy'))
h28v6_16 = (np.load('h28v06_16_train.npy'))
h29v5_16 = (np.load('h29v05_16_train.npy'))
h29v6_16 = (np.load('h29v06_16_train.npy'))
h31v6_16 = (np.load('h31v06_16_weathertrain.npy'))
pm_train_16 = np.load('localstations_pm2.5_valid.npy')

valid_batch = np.load('h31v06_weathertest.npy')
valid_sample = valid_batch.shape[0]

sample_per_valid = train_sample/sample_batch


test_data16= pred_generat(h28v5_16,h28v6_16,h29v5_16,h29v6_16,
                        h31v6_16,pm_train_16,
                            sample_batch,chan,chan2,chan3,time_t,time_step)


layer_name = 'concatenate_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

predictions_edit = intermediate_layer_model.predict_generator(test_data16,steps = sample_per_valid, verbose=1,workers=0)



print(predictions_edit.shape)
np.save('middlelayer_foryear16_4hr_4tile.npy',predictions_edit)
#np.save('prediction14_4hr_ver2.npy',pred_graph)