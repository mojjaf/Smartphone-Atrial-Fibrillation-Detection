# -*- coding: utf-8 -*-
"""ResNet_module.ipynb

Created on Wed Nov 13 16:11:12 2019

@author: Mojtaba Jafaritadi, Ph.D.
"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,LeakyReLU
from keras.models import Model, load_model
from keras import regularizers 
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform,he_normal
import scipy.misc
from matplotlib.pyplot import imshow
from keras.callbacks import  Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import Input, ZeroPadding1D, Dropout, LSTM, CuDNNLSTM, GRU, concatenate,Concatenate, Bidirectional,RepeatVector
from keras.layers.convolutional import Conv1D, AveragePooling1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l1,l2
from keras import optimizers
from keras.layers import TimeDistributed
import keras.backend as K

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block for RESNET
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=1,strides =1,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a')(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f,strides =1,padding='same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.25)(X)
    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c')(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    ### END CODE HERE ###
    
    return X

def maxpool_block_1(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=f,strides =s,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2a', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.15)(X)
    # Second component of main path ()
    X = Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    
    X_shortcut = Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '1', kernel_initializer = he_normal(seed=0))(X_shortcut)

    X_shortcut = MaxPooling1D(pool_size=s, padding='same',name = mp_name_base + '1')(X_shortcut)

    

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
def maxpool_block_2(X, f, filters, s, stage, block):
    """
    Implementation of the identity block for RESNET
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    mp_name_base = 'mp' + str(stage) + block + '_branch'
    # Retrieve Filters
    F1, F2 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]

    # First component of main path
    X = BatchNormalization(name = bn_name_base + '2a')(X)
   # X = Activation('relu')(X)    
    X = LeakyReLU(alpha=0.3)(X)    
    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F1, kernel_size=f,strides =s,padding='same',input_shape=(None,n_timesteps,n_features), name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
   # X = Activation('relu')(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = Dropout(0.15)(X)
    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F2, kernel_size=1, strides =1,padding='same', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=0))(X)
   

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(filters=F2, kernel_size=1, strides=1, name = conv_name_base + '1', kernel_initializer = he_normal(seed=0))(X_shortcut)

    X_shortcut = MaxPooling1D(pool_size=s, padding='same',name = mp_name_base + '1')(X_shortcut)

   # print(X_shortcut.shape)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut,])
    #X = Activation('relu')(X)
    X = LeakyReLU(alpha=0.3)(X)
    
    return X
def convolutional_block(X, f, filters, stage, block,s=2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    #n_timesteps, n_features = X.shape[1], X.shape[2]
    # Save the input value
    X_shortcut = X
    n_timesteps, n_features = X.shape[1], X.shape[2]


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv1D(filters=F1, kernel_size=f,strides = s, padding='same', input_shape=(None,n_timesteps,n_features),name = conv_name_base + '2a', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)

    # Second component of main path (≈3 lines)
    X = Conv1D(filters=F2, kernel_size=f,strides = 1, padding='same', name = conv_name_base + '2b', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)
  

    # Third component of main path (≈2 lines)
    X = Conv1D(filters=F3, kernel_size=1,strides = 1,padding='same', name = conv_name_base + '2c', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = bn_name_base + '2c')(X)
    X = Activation('relu')(X)

    X = Dropout(0.3)(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv1D(filters=F3, kernel_size=1,strides = s,padding='same', name = conv_name_base + '1', kernel_initializer = he_normal(seed=0))(X_shortcut)
   # print(X_shortcut.shape)
    X_shortcut = BatchNormalization(name = bn_name_base + '1')(X_shortcut)
    #X_shortcut = Dropout(0.25)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = Activation('relu')(X)
        
    return X

def DenseResNet1D(input_shape = (2048,6), classes = 3):
    """
    Implementation of the popular ResNet5 and DenseNet

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    input1 = Input(input_shape)
    input2 = Input(input_shape)
    X_input = Concatenate()([input1, input2])
    #X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding1D(0)(X_input)
    
    # Stage 0
    
    X = Conv1D(filters=4, kernel_size=3,padding='same', name = 'conv1', kernel_initializer = he_normal(seed=0))(X)
    X = BatchNormalization(name = 'bn_conv1')(X)
    X = Dropout(0.15)(X)
    X = LeakyReLU(alpha=0.3)(X)

   # X = Activation('relu')(X)
    
    # Stage 1
    stage=1
    block_conv='conv_dense_a'
    block_mp='mp_dense_a'
    X = maxpool_block_1(X, f = 3, filters = [4, 8], s=1,stage = 1, block='a')

    X_shortcut_1=X

    # Stage 2 
    stage=2
    block_conv='conv_dense_b'
    block_mp='mp_dense_b'
    X = maxpool_block_2(X, 3, filters = [8, 16], s=2, stage=2, block='a')
    X = maxpool_block_2(X, 3, filters = [8, 16], s=1, stage=2, block='b')
    X = maxpool_block_2(X, 3, filters = [8, 16], s=2, stage=2, block='c')

    X_shortcut_1 = Conv1D(filters=16, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X = layers.Add()([X, X_shortcut_1])
    X = Dropout(0.15)(X)


    X_shortcut_2=X

    # Stage 3
    stage=3
    block_conv='conv_dense_c'
    block_mp='mp_dense_c'
    X = maxpool_block_2(X, 3, filters = [16, 32],s=1, stage=3, block='a')
    X = maxpool_block_2(X, 3, filters = [16, 32],s=2, stage=3, block='b')
    X = maxpool_block_2(X, 3, filters = [16, 32],s=1, stage=3, block='c')

    X_shortcut_1 = Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)
    X_shortcut_2 = Conv1D(filters=32, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X = layers.Add()([X, X_shortcut_1,X_shortcut_2])
    X = Dropout(0.15)(X)

    X_shortcut_3=X

    # Stage 4
    stage=4
    block_conv='conv_dense_d'
    block_mp='mp_dense_d'    
    X = maxpool_block_2(X, 3, filters = [32, 64],s=2, stage=4, block='a')
    X = maxpool_block_2(X, 3, filters = [32, 64],s=1, stage=4, block='b')
    X = maxpool_block_2(X, 3, filters = [32, 64],s=2, stage=4, block='c')

    X_shortcut_1 = Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = Conv1D(filters=64, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X = layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3])
    X = Dropout(0.15)(X)
    X_shortcut_4=X

    # Stage 5
    stage=5
    block_conv='conv_dense_e'
    block_mp='mp_dense_e'
    X = maxpool_block_2(X, 3, filters = [64, 128],s=1, stage=5, block='a')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=2, stage=5, block='b')
    X = maxpool_block_2(X, 3, filters = [64, 128],s=1, stage=5, block='c')

    X_shortcut_1 = Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = Conv1D(filters=128, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = MaxPooling1D(pool_size=2, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X = layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4])
    X = Dropout(0.15)(X)
    X_shortcut_5=X

    # Stage 6
    stage=6
    block_conv='conv_dense_f'
    block_mp='mp_dense_f'
    X = maxpool_block_2(X, 3, filters = [128, 256],s=2, stage=6, block='a')
    X = maxpool_block_2(X, 3, filters = [128, 256],s=1, stage=6, block='b')
    X = maxpool_block_2(X, 3, filters = [128, 256],s=2, stage=6, block='c')

    X_shortcut_1 = Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '1', kernel_initializer = he_normal(seed=0))(X_shortcut_1)
    X_shortcut_1 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '1')(X_shortcut_1)

    X_shortcut_2 = Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '2', kernel_initializer = he_normal(seed=0))(X_shortcut_2)
    X_shortcut_2 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '2')(X_shortcut_2)

    X_shortcut_3 = Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '3', kernel_initializer = he_normal(seed=0))(X_shortcut_3)
    X_shortcut_3 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '3')(X_shortcut_3)

    X_shortcut_4 = Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '4', kernel_initializer = he_normal(seed=0))(X_shortcut_4)
    X_shortcut_4 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '4')(X_shortcut_4)

    X_shortcut_5 = Conv1D(filters=256, kernel_size=1, strides=1, name = block_conv + str(stage)  + '5', kernel_initializer = he_normal(seed=0))(X_shortcut_5)
    X_shortcut_5 = MaxPooling1D(pool_size=4, padding='same',name = block_mp+str(stage) + '5')(X_shortcut_5)

    X = layers.Add()([X, X_shortcut_1,X_shortcut_2,X_shortcut_3,X_shortcut_4,X_shortcut_5])

    X = BatchNormalization(name = 'bn_final')(X)
    X = Dropout(0.15)(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(pool_size=4,padding='same', name='max_pool_final')(X)

### LSTM 
    X = LSTM(units=32, return_sequences=True)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.3)(X)
    X = LSTM(units=32)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.3)(X)


    X=Dense(64, activation='relu',kernel_regularizer=l2(0.001))(X)
    X = Dropout(0.15)(X)

    X=Dense(32, activation='relu',kernel_regularizer=l2(0.001))(X)
    X = Dropout(0.15)(X)

    X=Dense(16, activation='relu',kernel_regularizer=l2(0.001))(X)
    X = Dropout(0.15)(X)

    X=Dense(8, activation='relu',kernel_regularizer=l2(0.001))(X)
    X = Dropout(0.15)(X)
    

    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    
    # Create model
    model = Model(inputs=[input1, input2], outputs=X,name='DenseResNet1D')


    return model

model = DenseResNet1D(input_shape = (2048, 6), classes = 3)