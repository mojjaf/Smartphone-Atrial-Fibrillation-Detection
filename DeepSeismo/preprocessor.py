# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:27:39 2019

@author: mkaist
"""
import numpy as np
from scipy import interpolate
import scipy.signal as signal

FS_NEW = 200 #resampled freq is 200
CS = 10 #samples cut from the beginning

def create_algorithm_input(a_raw, g_raw):

    a_raw = np.array(a_raw)[CS:,:]
    g_raw = np.array(g_raw)[CS:,:]    

    a_time = a_raw[:,3] - a_raw[:,3][1]
    g_time = g_raw[:,3] - g_raw[:,3][1]

    dt = 1e9/FS_NEW  #ns --> 5ms (200 Hz)
    resampled_index_acc = np.arange(0, a_time[-1], dt)
    resampled_index_gyr = np.arange(0, g_time[-1], dt)
    cutval = min(len(resampled_index_acc), len(resampled_index_gyr)) #take shortest from acc and gyro

    spl_accx=interpolate.interp1d(a_time, a_raw[:,0], kind='cubic')
    spl_accy=interpolate.interp1d(a_time, a_raw[:,1], kind='cubic')
    spl_accz=interpolate.interp1d(a_time, a_raw[:,2], kind='cubic')
    spl_gyrx=interpolate.interp1d(g_time, g_raw[:,0], kind='cubic')
    spl_gyry=interpolate.interp1d(g_time, g_raw[:,1], kind='cubic')
    spl_gyrz=interpolate.interp1d(g_time, g_raw[:,2], kind='cubic')
    
    ax = spl_accx(resampled_index_acc)[0:cutval]
    ay = spl_accy(resampled_index_acc)[0:cutval]
    az = spl_accz(resampled_index_acc)[0:cutval]
    gx = spl_gyrx(resampled_index_gyr)[0:cutval]
    gy = spl_gyry(resampled_index_gyr)[0:cutval]
    gz = spl_gyrz(resampled_index_gyr)[0:cutval]
  
    assert(len(ax)==len(ay)==len(az)==len(gx)==len(gy)==len(gx))
    
    return { 'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz }


def butter_filter(data, lf=1, hf=40, fs=FS_NEW, order=2):
    wbut = [2*lf/fs, 2*hf/fs] 
    bbut, abut = signal.butter(order, wbut, btype = 'bandpass') 
    
    if type(data)==dict:    
        for key in data:    
            data[key] = signal.filtfilt(bbut, abut, data[key]) 
        return data
    else: 
        return signal.filtfilt(bbut, abut, data)



def resample_data(a_raw, g_raw, sampling_freq = FS_NEW):
    """
    Created on Wed Sep 18 10:42:54 2019
    
    resampling smartphone data using Fourier method (scipy resample function)

    """
    
    from scipy import signal
    
    ################################################### discarding the forst 10 samples, what is the duration in ns 
    
    duration_acc_ns = a_raw[-1,3] - a_raw[10,3]
    duration_gyro_ns = g_raw[-1,3] - g_raw[10,3]
    
    ################################################### transform the duration in ns to s 
    
    duration_acc = duration_acc_ns * 1e-9
    duration_gyro = duration_gyro_ns * 1e-9
    
    minimum_duration = min([duration_acc, duration_gyro])
    
#    print(minimum_duration)
    
    ################################################### get the desired output after resampling 
    
    output_sig_length = int(minimum_duration * sampling_freq)
    
#    print(output_sig_length)
    
    ################################################## do the resampling
    
    ax = signal.resample(a_raw[:,0], output_sig_length)
    ay = signal.resample(a_raw[:,1], output_sig_length)
    az = signal.resample(a_raw[:,2], output_sig_length)
    
    gx = signal.resample(g_raw[:,0], output_sig_length)
    gy = signal.resample(g_raw[:,1], output_sig_length)
    gz = signal.resample(g_raw[:,2], output_sig_length)
    
    ################################################# return the output
    assert(len(ax)==len(ay)==len(az)==len(gx)==len(gy)==len(gx))
    
    return { 'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz }