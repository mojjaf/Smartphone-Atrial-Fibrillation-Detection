#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:31:01 2019

@author: Mojtaba Jafaritadi. Ph.D.
"""

from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
import numpy as np

###########################
def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_ax = CubicSpline(xx[:,0], yy[:,0])
    cs_ay = CubicSpline(xx[:,1], yy[:,1])
    cs_az = CubicSpline(xx[:,2], yy[:,2])
    cs_gx = CubicSpline(xx[:,3], yy[:,3])
    cs_gy = CubicSpline(xx[:,4], yy[:,4])
    cs_gz = CubicSpline(xx[:,5], yy[:,5])

    return np.array([cs_ax(x_range),cs_ay(x_range),cs_az(x_range),cs_gx(x_range),cs_gy(x_range),cs_gz(x_range)]).transpose()

def DA_MagWarp(X, sigma):
    return X * GenerateRandomCurves(X, sigma)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    tt_cum[:,3] = tt_cum[:,3]*t_scale[3]
    tt_cum[:,4] = tt_cum[:,4]*t_scale[4]
    tt_cum[:,5] = tt_cum[:,5]*t_scale[5]

  
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    X_new[:,3] = np.interp(x_range, tt_new[:,3], X[:,3])
    X_new[:,4] = np.interp(x_range, tt_new[:,4], X[:,4])
    X_new[:,5] = np.interp(x_range, tt_new[:,5], X[:,5])

    return X_new

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=3)
    angle = np.random.uniform(low=-np.pi, high=0.75*np.pi)
    acc_=np.matmul(X[:,0:3] , axangle2mat(axis,angle))
    gyr_=np.matmul(X[:,3:6], axangle2mat(axis,angle))
    X_new=np.column_stack((acc_,gyr_))
    return X_new

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)
    
    
def RandSampleTimesteps(X, nSample=1000):
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,3] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,4] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,5] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))

    tt[-1,:] = X.shape[0]-1
    return tt


def DA_RandSampling(X, nSample=1000):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    X_new[:,2] = np.interp(np.arange(X.shape[0]), tt[:,2], X[tt[:,2],2])
    return X_new



def augment_signals(input_data):
    outputdata=np.empty(input_data.shape, dtype=float, order='C')
    for i in range(input_data.shape[0]):
        X=input_data[i,:,:]
        X_augmented=DA_Rotation(DA_Permutation(X, nPerm=2))
        outputdata[i,:,:]=X_augmented
       
    return(outputdata)
        
        
    
    
    
    

