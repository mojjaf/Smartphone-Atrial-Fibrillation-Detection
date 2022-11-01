# -*- coding: utf-8 -*-
"""ResNet_module.ipynb

Created on Wed Nov 13 16:11:12 2019

@author: Mojtaba Jafaritadi, Ph.D.
"""
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


def plainseqnet(n_timesteps, n_features, classes = 9):
     # Define the input as a tensor with shape input_shape
    X_input = Input(shape=(n_timesteps, n_features), dtype='float32')
    # Zero-Padding
    X = ZeroPadding1D(0)(X_input)
        
    n_timesteps, n_features = X.shape[1], X.shape[2]

    #################################### build the model 

    
    X= Conv1D(filters=32, kernel_size=31, input_shape=(None,n_timesteps,n_features))(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)    
    BatchNormalization()(X)
    MaxPooling1D(pool_size = 2)(X)
    
    X = Conv1D(filters=32, kernel_size=21)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D(pool_size = 2)(X)
    
    X = Conv1D(filters=32, kernel_size=15)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D(pool_size = 2)(X)##### new architecture
    
    X = Conv1D(filters=32, kernel_size=11)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D(pool_size = 2)(X)
    
    X = Conv1D(filters=32, kernel_size=9)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)    
    X = BatchNormalization()(X)
    X = MaxPooling1D(pool_size = 2)(X) ###### new architecture 
    
    X = Conv1D(filters=32, kernel_size=7) (X)## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D(pool_size = 2)(X)
    
    X = Conv1D(filters=32, kernel_size=5)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D(pool_size = 2)(X)
#    
    X = Conv1D(filters=64, kernel_size=3)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = BatchNormalization()(X)

    X = LSTM(32, return_sequences=False)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Dropout(0.5)(X)

    X = Dense(32)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)

    X = Dense(32)(X) ## add kernel constraint , kernel_constraint=unit_norm()
    X = Activation('relu')(X)
    X = Dropout(0.5)(X)
    
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes))(X)
    
    # Create model
    #model = Model(inputs = X_input, outputs = X, name='ResNet1D')
    model = Model(inputs=X_input, outputs=X,name='plainseqnet')
    try:
            model = multi_gpu_model(model, gpus=4, cpu_relocation=True)
            print("Training on 4 GPUs")
    except:
            print("Training on 1 GPU/CPU")

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])


    return model
