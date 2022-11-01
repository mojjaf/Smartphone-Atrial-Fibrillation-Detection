#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import tensorflow as tf
import numpy as np
import pandas as pd 
import h5py
import os
from datetime import datetime, date, time, timezone
import matplotlib.pyplot as plt
from test_helpers import load_data_mat
import tensorflow as tf
print(tf.__version__)
import keras as K
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import plot_model
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import  Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler,LambdaCallback,EarlyStopping,CSVLogger
from preprocess_and_segmentation import preprocess_inputdata,segment_all_dict_data,reshape_segmented_arrays,plot_segment
from keras_utilities import evaluate_model,augmenting_func,plot_history,scheduler,lr_scheduler,summarize_predictions
from h5_helperfunctions import save_dict_to_hdf5, load_dict_from_hdf5
from SeismoNet import DenseResNet1D
from data_augmenter import augment_signals
from keras.utils import multi_gpu_model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

np.random.seed(1368)

############################# import and resample the train data 

input_train_AFib = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/traindata/AFib/AFib_train.h5')
input_train_SR = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/traindata/SR/SR_train.h5')
input_train_noise = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/traindata/Noise/noise_train.h5')

############################# 
#preprocess and filter the train data 
#############################  
AFib_train_org,AFib_train_env=preprocess_inputdata(input_train_AFib,label=1)
SR_train_org,SR_train_env=preprocess_inputdata(input_train_SR,label=2)
Noise_train_org,Noise_train_env=preprocess_inputdata(input_train_noise,label=3)

all_train_org={}

all_train_org.update(AFib_train_org)
all_train_org.update(SR_train_org)
all_train_org.update(Noise_train_org)

all_train_env={}

all_train_env.update(AFib_train_env)
all_train_env.update(SR_train_env)
all_train_env.update(Noise_train_env)

segment_length = 2048
segmented_data_org= segment_all_dict_data(data_dict = all_train_org, seg_width=segment_length, overlap_perc = 0.5)
segmented_data_env= segment_all_dict_data(data_dict = all_train_env, seg_width=segment_length, overlap_perc = 0.5)


X_org, y_org,ID_train_org = reshape_segmented_arrays(segmented_data_org, shuffle_IDs = False, 
                                    shuffle_segments = False, 
                                    outlier_rejection_flag = False, 
                                    segment_standardization_flag = True)
X_aug_rotperm=augment_signals(X_org)
X_org_aug=np.vstack((X_org,X_aug_rotperm))
y_org_ext=np.vstack((y_org,y_org))

 
X_env, y_env, ID_train_env = reshape_segmented_arrays(segmented_data_env, shuffle_IDs = False, 
                                    shuffle_segments = False, 
                                    outlier_rejection_flag = False, 
                                    segment_standardization_flag = True)
X_aug_rotperm_env=augment_signals(X_env)
X_env_aug=np.vstack((X_env,X_aug_rotperm_env))
#y_org_ext=np.vstack((y_org,y_org))
#X_org_aug=X_dev_env
#y_org_ext=y_env
############################# import and resample the VALIDATION data 

input_dev_AFib = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/devdata/AFib/AFib_valid.h5')
input_dev_SR = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/devdata/SR/SR_valid.h5')
input_dev_noise = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/devdata/Noise/noise_valid.h5')

############################# preprocess and filter the development data 
AFib_dev_org,AFib_dev_env=preprocess_inputdata(input_dev_AFib,label=1)
SR_dev_org,SR_dev_env=preprocess_inputdata(input_dev_SR,label=2)
Noise_dev_org,Noise_dev_env=preprocess_inputdata(input_dev_noise,label=3)

all_dev_org={}

all_dev_org.update(AFib_dev_org)
all_dev_org.update(SR_dev_org)
all_dev_org.update(Noise_dev_org)

all_dev_env={}

all_dev_env.update(AFib_dev_env)
all_dev_env.update(SR_dev_env)
all_dev_env.update(Noise_dev_env)

############################# segment and reshape

segmented_devdata_org= segment_all_dict_data(data_dict = all_dev_org, seg_width=segment_length, overlap_perc = 0.5)
segmented_devdata_env= segment_all_dict_data(data_dict = all_dev_env, seg_width=segment_length, overlap_perc = 0.5)

X_dev_org, y_dev_org,ID_dev_org = reshape_segmented_arrays(segmented_devdata_org, shuffle_IDs = False, 
                                    shuffle_segments = False, 
                                    outlier_rejection_flag = False, 
                                    segment_standardization_flag = True)
#X_augmented_dev=augment_signals(X_dev_org)
#X_org_aug_dev=np.vstack((X_dev_org,X_augmented_dev))
#y_devorg_ext=np.vstack((y_dev_org,y_dev_org))


X_dev_env, y_dev_env, ID_dev_env = reshape_segmented_arrays(segmented_devdata_env, shuffle_IDs = False, 
                                    shuffle_segments = False, 
                                    outlier_rejection_flag = False, 
                                    segment_standardization_flag = True)



######################################### TEST DATA
input_test_AFib = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/test_dataset/AFib_test.h5')
input_test_SR = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/test_dataset/SR_test.h5')
input_test_noise = load_dict_from_hdf5('/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/data/test_dataset/noise_test.h5')

############################# preprocess and filter the development data 
AFib_test_org,AFib_test_env=preprocess_inputdata(input_test_AFib,label=1)
SR_test_org,SR_test_env=preprocess_inputdata(input_test_SR,label=2)
Noise_test_org,Noise_test_env=preprocess_inputdata(input_test_noise,label=3)

all_test_org={}

all_test_org.update(AFib_test_org)
all_test_org.update(SR_test_org)
all_test_org.update(Noise_test_org)

all_test_env={}

all_test_env.update(AFib_test_env)
all_test_env.update(SR_test_env)
all_test_env.update(Noise_test_env)
############################# segment and reshape
segmented_testdata_org= segment_all_dict_data(data_dict = all_test_org, seg_width=segment_length, overlap_perc = 0.5)
segmented_testdata_env= segment_all_dict_data(data_dict = all_test_env, seg_width=segment_length, overlap_perc = 0.5)

X_test_org, y_test_org,ID_test_org = reshape_segmented_arrays(segmented_testdata_org, shuffle_IDs = False, 
                                    shuffle_segments = False, 
                                    outlier_rejection_flag = False, 
                                    segment_standardization_flag = True)

X_test_env, y_test_env, ID_test_env = reshape_segmented_arrays(segmented_testdata_env, shuffle_IDs = False, 
                                    shuffle_segments = False, 
                                    outlier_rejection_flag = False, 
                                    segment_standardization_flag = True)

#################################### classification using ResNet (Residual Networks on 1D data)


#model.summary()
X_train=X_org_aug
X_val=X_dev_org
X_test=X_test_org

X_train_env=X_env_aug
X_val_env=X_dev_env
X_test_env=X_test_env


enc = OneHotEncoder()
enc.fit(y_org_ext)
Y_train = enc.transform(y_org_ext).toarray()
Y_val = enc.transform(y_dev_org).toarray()
Y_test = enc.transform(y_test_org).toarray()

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of validation examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))

print ("X_val shape: " + str(X_val.shape))
print ("Y_val shape: " + str(Y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))



lr_reducer_loss = ReduceLROnPlateau(monitor = 'val_loss',factor=0.1,cooldown=1, patience=10) ### reducing the learning rate with a factor of 10 every 5 epochs
            
#lr_reducer_acc = ReduceLROnPlateau(monitor = 'val_accuracy' , factor=1e-4, mode = 'max', cooldown=1,patience=3, min_lr=0.5e-6)
            

######################################## 


filepath="/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/ResnetHannon_model_weights_multi.ckpt"

######################################## 


def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def run_experiment(repeats=10):
    output_directory = '/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/Final Simulation results/10secfused/' 
    #output_directory = '/scratch/project_2001856/jafarita/wrk/jafarita/AFIBDETECTOR/' 

	# repeat experiment
    scores = list()
    dict_of_results=dict()
    for r in range(repeats):
        model = DenseResNet1D(input_shape = (segment_length, 6), classes = 3)
        # Replicates `model` on 4 GPUs.
        # This assumes that your machine has 4 available GPUs.
        BATCH_SIZE=16
        NUM_GPU=2
        batch_num=BATCH_SIZE*NUM_GPU
        parallel_model = multi_gpu_model(model, gpus=NUM_GPU)
        parallel_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0, mode='auto')        
        checkpoint = ModelCheckpoint(os.path.join(output_directory , "model_checkpoint_iter_" + str(r) + ".ckpt"), monitor= 'val_acc' ,verbose=0, save_best_only=True, mode='max', save_weights_only=True)
        csv_logger = CSVLogger(os.path.join(output_directory , "model_historylog_iter_" + str(r) + ".csv"), append=False)
        parallel_model.fit([X_train,X_train_env], Y_train, validation_data= ([X_val,X_val_env], Y_val), epochs=200, batch_size= batch_num, callbacks=[csv_logger,checkpoint], shuffle = True, verbose=1)

        _, accuracy = parallel_model.evaluate([X_test,X_test_env], Y_test, batch_size=batch_num, verbose=0)
        print(accuracy)
        scores.append(accuracy)
        y_pred = parallel_model.predict([X_test,X_test_env])
        dict_of_results['iter_' + str(r)] = {}
        # accuracy: (tp + tn) / (p + n)
        dict_of_results['iter_' + str(r)]['accuracy'] = accuracy_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # precision tp / (tp + fp)
        dict_of_results['iter_' + str(r)]['precision_mic'] = precision_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='micro')
        dict_of_results['iter_' + str(r)]['precision_mac'] = precision_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='macro')
        dict_of_results['iter_' + str(r)]['precision_we'] = precision_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted')

        # recall: tp / (tp + fn)
        dict_of_results['iter_' + str(r)]['recall_mic'] = recall_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='micro')
        dict_of_results['iter_' + str(r)]['recall_mac'] = recall_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='macro')
        dict_of_results['iter_' + str(r)]['recall_we'] = recall_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted')

        # f1: 2 tp / (2 tp + fp + fn)
        dict_of_results['iter_' + str(r)]['f1_mic'] = f1_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='micro')
        dict_of_results['iter_' + str(r)]['f1_mac'] = f1_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='macro')
        dict_of_results['iter_' + str(r)]['f1_we'] = f1_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1),average='weighted')

        # kappa
        dict_of_results['iter_' + str(r)]['kappa'] = cohen_kappa_score(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
        # confusion matrix
        dict_of_results['iter_' + str(r)]['confusion matrix'] = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
        #saving the model weights for each iteration
        parallel_model.save_weights(os.path.join(output_directory , "model_weights_iter_" + str(r) + ".h5"))
        print("Saved model to disk")

    return(dict_of_results,scores)


    
# run the experiment
class_report,scores= run_experiment()
# summarize results
summarize_results(scores)
save_dict_to_hdf5(class_report,'AFib_classification_report_dict_10seconds_FUSEDinput.h5')
print("Saved confusion matrices to disk")