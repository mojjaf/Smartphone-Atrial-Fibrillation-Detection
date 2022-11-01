#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras as K
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from keras.callbacks import  Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input,Dense,Flatten,BatchNormalization, Activation, ZeroPadding1D, Add, Dropout, LSTM
from keras.layers.convolutional import Conv1D, AveragePooling1D, MaxPooling1D
from keras.models import Model, Sequential
from keras import optimizers
from keras.utils import plot_model
from keras.layers import TimeDistributed
from keras.utils import to_categorical

# In[ ]:
import math
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

def lr_scheduler(epoch, lr):
    decay_rate = 0.001
    decay_step = 0.10
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def scheduler(epoch):
  if epoch < 10:
    return 0.001
  else:
    return 0.001 * np.exp(0.1 * (10 - epoch))


# In[ ]:


def evaluate_model(train_X, train_y, val_X, val_y, my_activation = 'relu', my_baseline_loss_thresh = 0.4, 
                   my_baseline_metric_thresh =  0.99 , my_val_split = 0.15, early_stopping_flag = False, 
                   plot_model_flag= False,  epochs = 1 , batch_size = 32, my_verbose = 0, my_shuffle = True,
                   fit_flag = False, my_patience = 10 , my_loss = 'sparse_categorical_crossentropy'):
    
    
    if my_loss == 'sparse_categorical_crossentropy': 
        
        n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], len(np.unique(train_y))
    else: 
    
        n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_y.shape[1]
    
    
     # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 250
    train_X = train_X.reshape((train_X.shape[0], n_steps, n_length, n_features))
    val_X = val_X.reshape((val_X.shape[0], n_steps, n_length, n_features))     
    #################################### build the model 
    # define CNN LSTM model
    
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    #############
    '''
    
    model = Sequential()
    
    model.add(Conv1D(filters=32, kernel_size=31, input_shape=(n_timesteps,n_features))) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2))
    
    model.add(Conv1D(filters=32, kernel_size=21)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2))
    
    model.add(Conv1D(filters=32, kernel_size=15)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2)) ##### new architecture
    
    model.add(Conv1D(filters=32, kernel_size=11)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2))
    
    model.add(Conv1D(filters=32, kernel_size=9)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2)) ###### new architecture 
    
    model.add(Conv1D(filters=32, kernel_size=7)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2))
    
    model.add(Conv1D(filters=32, kernel_size=5)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size = 2))
#    
    model.add(Conv1D(filters=64, kernel_size=3)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation))
    model.add(BatchNormalization())
#    model.add(MaxPooling1D(pool_size = 2))
#    model.add(Dropout(0.5))

#    model.add(Flatten())
    model.add(LSTM(32, return_sequences=False)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Dropout(0.5))

    model.add(Dense(32)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation)) 
    model.add(Dropout(0.5))

    model.add(Dense(32)) ## add kernel constraint , kernel_constraint=unit_norm()
    model.add(Activation(my_activation)) 
    model.add(Dropout(0.5))
    
    model.add(Dense(n_outputs, activation='softmax'))
    '''
    ###################################### print the summary 

    print(model.summary())
    
    #####################################
    
    if plot_model_flag:
            plot_model(model , to_file = '/content/drive/My Drive/Colab Notebooks/AFib Detection/models/my_model.pdf', show_shapes=True, show_layer_names=True)
    
    ###################################### fit network
    if fit_flag: 
        
        ###################################### compile the model 
        
#        opt = optimizers.Adam( )
#        model.compile(loss=my_loss, optimizer=opt, metrics=['accuracy'])	
    
        my_learning_rate = lr_schedule(0)
        model.compile(loss=my_loss, optimizer=optimizers.Adam(learning_rate = my_learning_rate) , metrics=['accuracy'])	
 
        
        ##########
    
        if early_stopping_flag:
            
            ########################################################## call back class to terminate the classification at a threshold
            
            ################## callback for the maximum accuracy by threshold
            
            class TerminateOnBaselineACC(Callback):
                """Callback that terminates training when either acc or val_acc reaches a specified baseline
                """
                def __init__(self, monitor='acc', baseline=0.9):
                    super(TerminateOnBaselineACC, self).__init__()
                    self.monitor = monitor
                    self.baseline = baseline
            
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    metric = logs.get(self.monitor)
                    if metric is not None:
                        if metric >= self.baseline:
                            print('Epoch %d: Reached baseline, terminating training' % (epoch))
                            self.model.stop_training = True
           
            ################# callback for the minimum loss by threshold
            
#            class EarlyStoppingByLossVal(Callback):
#                def __init__(self, monitor='val_loss', value=0.00001):
#                    super(Callback, self).__init__()
#                    self.monitor = monitor
#                    self.value = value
#            
#                def on_epoch_end(self, epoch, logs={}):
#                    current = logs.get(self.monitor)
#                   
#                    if current is not None:
#                        if current < self.value:
#                            print("Epoch %05d: early stopping THR" % epoch)
#                            self.model.stop_training = True
                            
            ################ third callback for saving the best model if the other callbacks were not called
            
            filepath="/content/drive/My Drive/Colab Notebooks/AFib Detection/models/the_best_model.ckpt"
            checkpoint = ModelCheckpoint(filepath, monitor= 'val_accuracy' , verbose=1, save_best_only=True, mode='max')
            
            ################ 
            
            lr_scheduler = LearningRateScheduler(lr_schedule)

            lr_reducer_loss = ReduceLROnPlateau(factor=1e-4,
                                               cooldown=0,
                                               patience=5,
                                               min_lr=0.5e-6)
            
            lr_reducer_acc = ReduceLROnPlateau(monitor = 'val_accuracy' , 
                                               factor=1e-4,
                                               mode = 'max', 
                                               cooldown=0,
                                               patience=5,
                                               min_lr=0.5e-6)
            
            ######################################## 
            
            my_callbacks = [TerminateOnBaselineACC(monitor='val_accuracy', baseline=my_baseline_metric_thresh),
                            checkpoint,
                            lr_scheduler,
                            lr_reducer_loss]
            
#            my_callbacks = [TerminateOnBaselineACC(monitor='val_accuracy', baseline=my_baseline_metric_thresh),   #EarlyStoppingByLossVal(monitor='val_loss', value=my_baseline_loss_thresh),
#                            checkpoint ]
            
#            my_callbacks = [EarlyStoppingByLossVal(monitor=my_metric_to_monitor, value=my_baseline_thresh)]

#            early_stop = EarlyStopping(monitor=my_callbacks, patience=my_patience)
        
            history = model.fit(train_X, train_y, 
                                validation_data= (val_X, val_y),
                                epochs=epochs, 
                                batch_size=batch_size, 
                                shuffle = my_shuffle, 
                                callbacks = my_callbacks)
        
        ##################################
        else: 
            history = model.fit(train_X, train_y, 
                                validation_data= (val_X, val_y),
                                epochs=epochs, 
                                batch_size=batch_size, 
                                shuffle = my_shuffle)
        #########
        return(history, model)

    ####################################### evaluate model
    else: 
        return( model)


# In[ ]:


def plot_history(history):
  
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = np.arange(len(history.history['loss']))

  plt.figure()
  fig=plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.savefig('./learningcurves.png')
  #plt.show()
  


# In[ ]:


def summarize_predictions(my_model, X_test, y_test, ID_test):
    
    def one_hot_decoder(row):
        
        label = np.where(row == 1)[0].astype(int)
        
        return label
    
    ################################################# test the model
    
    pred_test_proba = my_model.predict(X_test, verbose = 0)

    ################################################# transform one hot encoded predictions to int
    
    int_pred = np.around(pred_test_proba).astype(int)
    
    pred_test_arr = pd.DataFrame(int_pred).idxmax(axis=1)      
    
    pred_test_arr = pred_test_arr + 1
    
    #################################################
    
    print('Confusion matrix before taking statistical mode of the predictiod labels: \n', 
          confusion_matrix(y_test.ravel(), pred_test_arr ) ) 
    
    ################################################# transform the data into a pandas data frame
    
    result_df = pd.DataFrame( { 'ID': ID_test.ravel(), 'true_label': y_test.ravel().astype(int), 'pred_label' : pred_test_arr} )

    ################################################# group by and summarize 
    
    mode_of_pred = result_df.groupby('ID').agg(lambda x:x.value_counts().index[0])
    
    print('Confusion matrix after taking statistical mode of the predictiod labels: \n', 
          confusion_matrix(mode_of_pred['true_label'], mode_of_pred['pred_label'] ))

    return((mode_of_pred , pred_test_arr , pred_test_proba))    


# In[1]:




def scale_data(trainX, testX, overlap = 0.75):
    
    from sklearn.preprocessing import StandardScaler
    
    #################################### remove overlap
    
    cut = int(trainX.shape[1] * (1-overlap))
    longX = trainX[:, 0:cut , :]

    #################################### flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))

    #################################### flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))

    #################################### standardize

    s = StandardScaler()
    ################# fit on training data
    s.fit(longX)	
    
    ################# apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    
    ################# reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    flatTestX = flatTestX.reshape((testX.shape))

    return flatTrainX, flatTestX

######################################################


# In[ ]:


def augmenting_func(multidim_data, labels):
    
    """
    This function balances the class distributions in the input data set.
    The data sets can be training or validation 
    """
    
    class_lab , class_size = np.unique(labels, return_counts = True)
    
    ###################################################
    
    list_of_multidim_data_aug = []
    list_of_labels_aug = []
    
    max_size = np.max(class_size)
    
    ###################################################
    
    for lab, siz in zip(class_lab, class_size):
        
#        print(lab)
#        print(siz)
        
        
        if siz < max_size:
            
            numbers_to_produce = max_size - siz
            
            X_lab = multidim_data[(labels == lab).ravel(),:,:]
            
            X_lab_produced = X_lab[np.random.choice(X_lab.shape[0], numbers_to_produce),:,:]
            
            noise_produced = np.random.normal(size = X_lab_produced.shape)*0.05 
    
            X_lab_augmented = np.concatenate( ( X_lab , noise_produced), axis = 0 )
    
            y_lab_augmented = np.repeat(lab, X_lab_augmented.shape[0] )
    
            y_lab_augmented = y_lab_augmented[..., np.newaxis]
            
        else: 

            X_lab_augmented = multidim_data[(labels == lab).ravel(),:,:].copy()
            
            y_lab_augmented = labels[(labels == lab).ravel()].copy()
            
            
#        print(X_lab_augmented.shape)   
#        print(y_lab_augmented.shape)
        
        list_of_multidim_data_aug.append(X_lab_augmented)
            
        list_of_labels_aug.append(y_lab_augmented)
      
#        list_of_multidim_data_aug = np.vstack( [list_of_multidim_data_aug, X_lab_augmented] )
#        list_of_labels_aug = np.vstack( [list_of_labels_aug, y_lab_augmented] )

    #####################################################
    
    list_of_multidim_data_aug = tuple(list_of_multidim_data_aug)
    list_of_labels_aug = tuple(list_of_labels_aug)
    
    #####################################################
    
    stacked_aug_multidim = np.concatenate(list_of_multidim_data_aug, axis = 0)
    
    stacked_aug_labels = np.concatenate(list_of_labels_aug , axis = 0)
    
    return( (stacked_aug_multidim,  stacked_aug_labels)) #,  list_of_labels_aug