#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:39:47 2019

@author: Mojtaba Jafaritadi, Ph.D.
"""

######################################################################

from sklearn import preprocessing
import numpy as np
from itertools import compress
from scipy import stats, signal
from artifact_remover import remove_artifacts
import os
from test_helpers import load_data_json
from peak_detector import PCA_from_data,enhance_envelope
from preprocessor import butter_filter
import copy 
import glob 
import pandas as pd 
import matplotlib.pyplot as plt
import time

######################################################################

    
def label_input_data(input_data,label):
    
    for key in input_data.keys():
        
        input_data[key]['label'] = label
    
    input_data.update( input_data )

    return(input_data)
    
###################################################################### preprocess the data
def preprocess_inputdata(input_data,label) : 
    input_data_cp=copy.deepcopy(input_data)
    
    for key in input_data.keys():
        dataset1 = copy.deepcopy(input_data[key])
        dataset2 = copy.deepcopy(input_data[key])
        

        dataset1 = butter_filter(dataset1, lf=10, hf=70, order=4) #pca needs 10 to 70
        dataset1 = enhance_envelope(dataset1)
#        signal_ = PCA_from_data(dataset1)  
        
        dataset2 = butter_filter(dataset2, lf=3, hf=20, order=4) #normal filtering on the original signals 3-40 Hz

#        dataset2['gz'] = butter_filter(signal_, lf=1, hf=4)
        input_data[key]=dataset2
        input_data_cp[key]=dataset1

        
    output_data_org =label_input_data(input_data,label)
    output_data_env =label_input_data(input_data_cp,label)

    return (output_data_org,output_data_env)

################################################################### segmentation function
def window_stack(a, win_width , overlap):
    if overlap == 0:
        return(np.vstack( [a[i:(i+win_width)] for i in np.arange(0,(len(a)-win_width+1),win_width).astype(int)]  ) ) 
    else:
        stride = 1-overlap
        return(np.vstack( [a[i:(i+win_width)] for i in np.arange(0,(len(a)-win_width+1),win_width*stride).astype(int)]   ) )
 
    
    
################################################################### apply segmentation function to all the data
    
def remove_short_signal(input_dict, key):
    copy_dict = dict(input_dict)
    del copy_dict[key]
    return copy_dict

################################################################### apply segmentation function to all the data
def segmenting_data(dict_of_data, seg_width, overlap_perc):
    
    segmented_signals = {} 
    ignore_key=['label']
    for key in dict_of_data.keys() - ignore_key:

        segmented_signals[key] = window_stack(dict_of_data[key], seg_width , overlap = overlap_perc) 
                
        ############################################ add the label to the dict 
    

    segmented_signals['label'] = np.repeat(dict_of_data['label'],len(segmented_signals['az']))
    
    return segmented_signals
    
def segment_all_dict_data(data_dict, seg_width, overlap_perc):

    """
    This function abstracts all the segmenting and reshaping tasks that are performed 
    by the segmenting_data and reshape_segmented_arrays functions
    """

    segmented_dict_of_data=   {}
    ################################################## segment the signals into smaller pieces and put all reshaped data into a mother dictionary
    for key in data_dict.keys():
        if (data_dict[key]['ax'].shape[0] > seg_width):
           # print('##########  measurement {0} #############'.format(key))
            segmented_dict_of_data[key] = segmenting_data(data_dict[key], seg_width, overlap_perc)
        
    ################################################## reshape the data to make a 6-channel segment 
    return(segmented_dict_of_data)
    
    
#    

#    return(X,y,ID)
    
def mad(my_segment, theta = 10):
    
    my_median = np.median(my_segment)
    
    ########################### find the outliers 
    
    MedianAD = theta * (np.median(np.abs(my_segment - my_median ))) 
    
    MedianAD_flag = np.abs(my_segment - my_median ) 
    
    outliers = np.where(MedianAD_flag > MedianAD , 1, 0)
    
    ########################## get sign of the data 
    
    sign_of_my_segment = np.where(my_segment > 0, 1, -1 )
    
    ########################## replace the ones positive with my_median and the negative ones with -my_median
    
    cleaned_segment = my_segment.copy()
    
    outliers_to_replace = outliers*sign_of_my_segment
    
    cleaned_segment[np.where(outliers_to_replace == +1)] = abs(MedianAD)
             
    cleaned_segment[np.where(outliers_to_replace == -1)] = -abs(MedianAD)

    ######################### 
    
    return cleaned_segment   

    
def reshape_segmented_arrays(input_dict, shuffle_IDs = True, shuffle_segments = True, outlier_rejection_flag = True, segment_standardization_flag = True):
    
    from random import shuffle  
    
    #########################################
    
    list_of_swapped_stack = []
    
    list_of_ID_arrays = []
    
    list_of_label_arrays = []
    
    #########################################
    
#    for ID, dict_data in input_dict.items():
    for key in input_dict.keys():
        
        #print('############### THE ID IS : {0} ###############'.format(key))
              
        ##################################### list of the matrices of segmented data in 6 channel
        dict_data=input_dict[key]
        ID=key
        
        data_list = [v for k,v in dict_data.items() if k != 'label']
    
        ##################################### stacking all the data into one array
        
        data_stacked_array = np.stack(data_list, axis = 0)
        
        ##################################### outlier rejection by 5 and 95th percentile at each segment
        
        if outlier_rejection_flag:
            
#            data_stacked_array = outlier_rejection(data_stacked_array)
            
            data_stacked_array = np.apply_along_axis(mad, 2, data_stacked_array) 
            
        ##################################### shuffle the segments in the data_stacked_array cleaned
        
        if shuffle_segments: 
            
            random_indices = np.random.randint(0, data_stacked_array.shape[1], data_stacked_array.shape[1] )
        
            data_stacked_array = data_stacked_array[:, random_indices, :] 
        
        ##################################### swap the axes 
        
        swaped_stack = np.swapaxes ( np.swapaxes(data_stacked_array, 0 ,2) , 0 ,1)
        
        #####################################
        
        ID_for_segments = np.repeat (ID, swaped_stack.shape[0])
        
        label_for_segments = dict_data['label']
        
        #################################### append to their corresponding lists 
        
        list_of_swapped_stack.append( swaped_stack )

        list_of_ID_arrays.append(ID_for_segments)
        
        list_of_label_arrays.append(label_for_segments)
        
#        print(swaped_stack.shape)
    
    ################################### shuffle the order of subjects in every list 
    
    if shuffle_IDs: 
        
        ######################## generate random indices
        
        perm = list(range(len(list_of_ID_arrays)))
        shuffle(perm)
        
        #print(perm)
        
        ######################## rearrange the lists 
        
        list_of_swapped_stack = [list_of_swapped_stack[index] for index in perm]
        list_of_ID_arrays = [list_of_ID_arrays[index] for index in perm]
        list_of_label_arrays = [list_of_label_arrays[index] for index in perm]

    ################################### transform the lists into numpy arrays by stacking along first axis
    
    array_of_segments = np.concatenate(list_of_swapped_stack , axis = 0)
    
    array_of_IDs = np.concatenate(list_of_ID_arrays, axis = 0)[:, np.newaxis]

    array_of_labels = np.concatenate(list_of_label_arrays, axis = 0)[:, np.newaxis]
    
    ################################# normalize every segemnt 
    
    if segment_standardization_flag : 
        
        def segment_standardization(my_segment):
            
            from sklearn.preprocessing import StandardScaler
            
            ################# 
            s = StandardScaler()
        
            ################# fit on training data
        
            normalized_segment = s.fit_transform(my_segment[:,np.newaxis])	
            
            #############
            return(normalized_segment.ravel())
        
        ############################
        
        array_of_segments = np.apply_along_axis(segment_standardization, 1, array_of_segments)
                            
                            
    ################################# print the shapes
    
    print('shape of the array of segments is :', array_of_segments.shape )
    
    print('shape of the array of IDs is :', array_of_IDs.shape )
    
    print('shape of the array of labels is :', array_of_labels.shape )
        
        ##################################
        
    return(array_of_segments, array_of_labels , array_of_IDs)
    
    
def plot_segment(inputarray,seg_indx,axis1,axis2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(inputarray[seg_indx,:,axis1:axis2])
    plt.show()
    return(fig)