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

start_time = time.time()

n_channels = 1
n_classes = 18
optim = Adam(lr = 0.00001, decay = 0.00001)
sample_batch = 10
time_step = 1
time_step2 = 4
number_stations = 18
        
        
def eval_metrics_on(predictions,labels):
    '''assuming this is a regression task; labels are continuous-valued floats
    
    returns most regression-related scores for the given predictions/targets as a dictionary:
    
        r2, mean_abs_error, mse, rmse, median_absolute_error, explained_variance_score
    '''
    #if len(labels[0])==2: #labels is list of data/labels pairs
        #labels = np.concatenate([l[1] for l in labels])
    #predictions = predictions[:,0]
    prdictions = np.array(predictions)
    labels = np.array(labels)
    
    r2                       = metrics.r2_score(labels, predictions)
    mean_abs_error           = np.abs(predictions - labels).mean()
    mse                      = ((predictions - labels)**2).mean()
    rmse                     = np.sqrt(mse)
    #median_absolute_error    = metrics.median_absolute_error(labels, predictions) # robust to outliers
    explained_variance_score = metrics.explained_variance_score(labels, predictions) # best score = 1, lower is worse
    return {'r2':r2, 'mean_abs_error':mean_abs_error, 'mse':mse, 'rmse':rmse, 
    'explained_variance_score':explained_variance_score}


def predict_graph(true_val,predict_val):
    for sn in range(number_stations):
        plt.figure(figsize=(6,3))
        plt.figure(sn+1)
        plt.title('_next_4hour station = ' + str(sn+1))
        plt.plot(true_val[:,sn::number_stations], c='g', label= 'gt')
        plt.plot(predict_val[:,sn::number_stations], c='r', label= 'pred')
        plt.legend(loc='best',prop={'size': 5})
        plt.legend(loc='best')
        name = str("firstgroup_stat_") + str(sn) + str(".png")
        #plt.savefig(name)
        plt.show()


def gui_generator(tile28v05,tile28v06,tile29v05,tile29v06,tile31v06,pm,y_lst,batch_size,num_channels,
                  num_channels_2,num_channels_3,
                  time,time_label,cover_hr_in=1):
    

    while True:
        #x_shape_1 = (batch_size,num_channels,time_label,600,1200)
        x_shape_1 = (batch_size,time_label,num_channels,300,300)
        
        #x_shape_2 = (batch_size,num_channels,time_label,650,600)
        x_shape_2 = (batch_size,time_label,num_channels,300,300)
        
        x_shape_3 = (batch_size,time,num_channels_2,2,7)
        
        x_shape_4 = (batch_size,time_label,num_channels_3,4,6)
        
        y_shape = (batch_size,time,18)
        
        
        h28v05 = np.zeros(shape=x_shape_1)
        h28v06 = np.zeros(shape=x_shape_1)
        
        h29v05 = np.zeros(shape=x_shape_2)
        h29v06 = np.zeros(shape=x_shape_2)
        
        h31v06 = np.zeros(shape=x_shape_3)
        h32v06 = np.zeros(shape=x_shape_4)
        pm_batch = np.zeros(shape=y_shape)
        
    
        y_batch = np.zeros(shape=y_shape)
        
        end = y_lst.shape[0]
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
                
                y_batch[seq_idx] = y_lst[idx:idx+cover_hr_in][0]
               

            
            yield ([h28v05,h28v06,h29v05,h29v06,h31v06,pm_batch],y_batch)
            
            
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
        
        y_shape = (batch_size,time,18)
       
        
        
        h28v05 = np.zeros(shape=x_shape_1)
        h28v06 = np.zeros(shape=x_shape_1)
        
        h29v05 = np.zeros(shape=x_shape_2)
        h29v06 = np.zeros(shape=x_shape_2)
        
        h31v06 = np.zeros(shape=x_shape_3)
        #h32v06 = np.zeros(shape=x_shape_4)
        pm_batch = np.zeros(shape=y_shape)
        
        
        y_batch = np.zeros(shape=y_shape)
        
        end = pm.shape[0]
        
        
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

            

def mean_absolute_percentage_error2(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def fn_model_schw3(input_data):
    model_one_conv_1 = AveragePooling3D(pool_size=(1,2,2), strides=(1,2,2),padding='same', data_format='channels_first')(input_data)
    #model_one_conv_1 = AveragePooling3D(pool_size=(1,2,2), strides=(1,2,2),padding='same', data_format='channels_first')(model_one_conv_1)
    model_one_conv_1 = TimeDistributed(Conv2D(32, (3,3),strides =(2,2), activation='relu',data_format='channels_first'))(model_one_conv_1)
    #model_one_conv_1 = TimeDistributed(AveragePooling2D(pool_size=(2,4), strides=(1,1),padding='same',data_format='channels_first'))(model_one_conv_1)
    model_one_conv_1 = TimeDistributed(Conv2D(32, (3,3) ,activation='relu',padding = 'same',data_format='channels_first'))(model_one_conv_1)
    
    model_one_conv_1 = TimeDistributed(Dropout(rate = 0.3))( model_one_conv_1)

    model_one_conv_1 = ConvLSTM2D(32, (3,3),padding='same', activation='relu',kernel_regularizer=L1L2(l1 = 0.01,l2=0.01),return_sequences=True)(model_one_conv_1)
    #model_one_conv_1 = AveragePooling3D(pool_size=(1,3,3), strides=(1,2,2),padding='same',data_format='channels_first')(model_one_conv_1)
    model_one_conv_1 = BatchNormalization()(model_one_conv_1)
    #model_one_conv_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2),padding='same')(model_one_conv_1)
    model_one_conv_1 = ConvLSTM2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=L1L2(l1 = 0.01,l2=0.01),return_sequences=True)(model_one_conv_1)
    
    model_one_conv_1 = AveragePooling3D(pool_size=(1,10,10), strides=(1,10,10),padding='same',data_format='channels_first')(model_one_conv_1)
    
    return model_one_conv_1
    
    
def weather_model_schw(input_data):
    #PLEASE USE TRANSPOSE AND SEE
    model_one_conv_1 = ConvLSTM2D(64, (1, 3),padding='same', activation='relu',kernel_regularizer=L1L2(l1 = 0.01,l2=0.01),return_sequences=True)(input_data)
    
    model_one_conv_1 = BatchNormalization()(model_one_conv_1)
    
    model_one_conv_1 = ConvLSTM2D(32, (1, 3), padding='same', activation='relu',kernel_regularizer=L1L2(l1 = 0.01,l2=0.01), return_sequences=True)(model_one_conv_1)
    
    return model_one_conv_1


def pm25_model(input_data):
    #model_one_conv_1 = LSTM(32, activation='relu',return_sequences=True)(input_data)
    model_one_conv_1 = LSTM(32, activation='relu',dropout = 0.3,recurrent_dropout=0.3,return_sequences=True)(input_data)
    return model_one_conv_1

def temporal_model(input_data):
    model_one_conv_1 = LSTM(128, activation='relu',dropout = 0.3,recurrent_dropout=0.3,return_sequences=True)(input_data)
    #model_one_conv_1 = LSTM(64, activation='relu',dropout = 0.3,recurrent_dropout=0.3,return_sequences=True)(model_one_conv_1)
    return model_one_conv_1
    


# Model 1 tile 1
data_shape = (1,1,300,300)
in1 = Input(shape=data_shape)
model1 = fn_model_schw3(in1)

# Model 2 for tile 2
data_shape1 = (1,1,300,300)
in2 = Input(shape=data_shape1)
model2 = fn_model_schw3(in2)

# Model 3 for tile 3
data_shape2 = (1,1,300,300)
in3 = Input(shape=data_shape2)
model3 = fn_model_schw3(in3)

# Model 4 for tile 4
data_shape3 = (1,1,300,300)
in4 = Input(shape=data_shape3)
model4 = fn_model_schw3(in4)




#remote weather model
weath_data_shape = (4,148,2,7)
in10 = Input(shape=weath_data_shape, name='weath_input')
model10 = weather_model_schw(in10)



all_batch = np.load('h31v06_weathertrain.npy')[:-2]
train_sample = all_batch.shape[0]
valid_batch = np.load('h31v06_weathertest.npy')
valid_sample = valid_batch.shape[0]

#
chan = 1
chan2 = 148
chan3 = 2
time_t = 4
sample_per_epoch = train_sample/sample_batch
sample_per_valid = valid_sample/sample_batch

#input for PM2.5 as feature
data_shape = (18,)
in12 = Input(shape=data_shape)
#model12 = pm25_model(in12)


model_final1 = Flatten()(model1)
model_final2 = Flatten()(model2)
model_final3 = Flatten()(model3)
model_final4 = Flatten()(model4)



model_final_concat_allAOD = concatenate([model_final1,model_final2,model_final3,model_final4])
model12 = RepeatVector(time_step2)(in12)


weather_model = TimeDistributed(Flatten())(model10)

model_final_concat = RepeatVector(time_step2)(model_final_concat_allAOD)
model_final_concat = concatenate([model_final_concat,weather_model])

#model_final_concat = concatenate([model_final_concat,model12])
model_final_concat = concatenate([model_final_concat,model12])


model_final_concat = TimeDistributed(Dense(32, activation='relu',bias_regularizer=L1L2(l1 = 0.01,l2=0.01)))(model_final_concat)
model_final_concat = TimeDistributed(Dropout(rate = 0.5))(model_final_concat)
model_final_concat = TimeDistributed(Dense(32, activation='relu',bias_regularizer=L1L2(l1 = 0.01,l2=0.01)))(model_final_concat)
model_final_concat = TimeDistributed(Dropout(rate=0.5))(model_final_concat)

model_final_concat = TimeDistributed(Dense(n_classes))(model_final_concat)


#model = Model(inputs=[in2,in4,in10,in12],outputs=model_final_concat)
model = Model(inputs=[in1,in2,in3,in4,in10,in12],outputs=model_final_concat)

model.summary()

NB_EPOCH = 550

model.compile(optimizer = optim, loss = 'mse', metrics = ['mse'])

keras_callback = [EarlyStopping(monitor='val_mse', patience = 80, mode='min'),
                 ModelCheckpoint('next4hr_4tiles_18stations_ver.h5', monitor='val_mse', 
                                 save_best_only=True,mode='min', verbose = 1)]

                            
history = model.fit_generator(gui_generator(np.load('h28v05_train.npy')[:-2],np.load('h28v06_train.npy')[:-2],
                                            np.load('h29v05_train.npy')[:-2],np.load('h29v06_train.npy')[:-2],
                                            np.load('h31v06_weathertrain.npy')[:-2],np.load('localstations_pm2.5_train.npy')[:-2],
                                            np.load('localstations_label_train.npy')[:-2],sample_batch,chan,chan2,chan3,time_t,time_step),
                                              steps_per_epoch = sample_per_epoch,
                                            epochs = NB_EPOCH,workers=0,shuffle = False,
                              validation_data = gui_generator(np.load('h28v05_test.npy'),
                                                              np.load('h28v06_test.npy'),
                                                np.load('h29v05_test.npy'),np.load('h29v06_test.npy'),
                                            np.load('h31v06_weathertest.npy'),np.load('localstations_pm2.5_valid.npy'),
                                                              np.load('localstation_label_valid.npy'),
                                            sample_batch,chan,chan2,chan3,time_t,time_step),
                                            validation_steps = sample_per_valid,verbose = 1,callbacks=keras_callback) 

                 
                                                        
#save model train history
np.save('history_n4hr_stri.npy',history.history)                         
                            

#save model
file_name = 'next4hr_stri_model.h5'
model.save(file_name)


test_data = pred_generat(np.load('h28v05_test.npy'),np.load('h28v06_test.npy'),
                         np.load('h29v05_test.npy'),np.load('h29v06_test.npy'),
                        np.load('h31v06_weathertest.npy'),np.load('localstations_pm2.5_valid.npy'),
                            sample_batch,chan,chan2,chan3,time_t,time_step)
        

predictions_edit = model.predict_generator(test_data,steps = sample_per_valid, verbose=1,workers=0)
print(len(predictions_edit))
predictions = predictions_edit.reshape(predictions_edit.shape[0]*predictions_edit.shape[1],predictions_edit.shape[2])
print(predictions_edit.shape)
y_label = np.load('localstation_label_valid.npy')
y_label = y_label.reshape(y_label.shape[0]*y_label.shape[1],y_label.shape[2])
print(y_label.shape,predictions.shape)

eval_results = eval_metrics_on(predictions,y_label)
print('this is evaluation results for r2,mean_abs_error,mse,rmse,expl_var_score ',eval_results)

#observe the graph between prediction and true value of PM2.5 for last hour
y_label = np.load('localstation_label_valid.npy')
y_true = y_label[:,3:4,:]
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
    t2 = y_label[:,i:i+1,:]
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
end_time = time.time()
print("Total execution time: {}".format(end_time - start_time))