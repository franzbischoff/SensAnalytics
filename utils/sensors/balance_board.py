import numpy as np
import pandas as pd
from scipy import signal
from utils.sensors.balance_board_helpers import featureHandler as fh

def _decimate(data,downsampling_factor):
    """The working portion of the decimate function. The downsampling_factor should be either an int of <= 10, or a list of integers <= 10, representing a cascading decimation"""
    
    if isinstance(downsampling_factor,int):
        if downsampling_factor > 10:
            print('Make sure the downsampling factor is less than 10. If you want it to be more than 10, cascade it and present it as a list. E.g., [10,10] gives a factor of 100')
            return
        else:
            data = signal.decimate(data,downsampling_factor)
    
    if isinstance(downsampling_factor,list):
        for factor in downsampling_factor:
            if factor > 10:
                print('Make sure the downsampling factor is less than 10. If you want it to be more than 10, cascade it and present it as a list. E.g., [10,10] gives a factor of 100')
                return
            else:    
                data = signal.decimate(data,factor)
                
    return data

def decimate(data,downsampling_factor):
    """Decimates (low pass filter and then downsamples) the signal. This decimates only the Fz, Mx, My, CoPx, and CoPy. That is, it does not decimate t, but rather simply downsamples it"""
    
    CoPx = _decimate(data[4],downsampling_factor)
    CoPy = _decimate(data[5],downsampling_factor)
    Fz = _decimate(data[1],downsampling_factor)
    Mx = _decimate(data[2],downsampling_factor)
    My = _decimate(data[3],downsampling_factor)
    
    t = []
    for i,ts in enumerate(data[0]): 
        if i % np.prod(downsampling_factor) == 0: t.append(ts)
    
    return np.array(t),Fz,Mx,My,CoPx,CoPy

def process_Ge(data, lowpass_cutoff, lowpass_order, decimate_order, demean=False, scale=1):
    """ Implementation of Wenbo Ge.
    Preprocess the data. This assumes the data hasnt been played with yet, i.e., fs = 1000
    A scale=1 means the units are in meters"""

    #If the data is 120 seconds, cut to 90 by removing last 30 seconds
    data = [d[:90000] for d in data]
    data[4] = data[4] * scale
    data[5] = data[5] * scale
    
    #Low pass filter
    sos = signal.butter(lowpass_order, lowpass_cutoff, 'lowpass', fs=1000, output='sos')
    data = [data[0]] + [signal.sosfilt(sos,d) for d in data[1:]]
    
    #Remove first and last 15 seconds, resulting in a 60 second sample
    #This also removes artifacts in the butterworth filter, as it assumes a start from 0, 0, which is not the case
    data = [d[15000:75000] for d in data]
    
    #Recenters the data if needed, but not t and Fz
    if demean: data = [data[0], data[1]] + [d - d.mean() for d in data[2:]]
    
    #Adjust time so that it starts from 0
    data[0] = data[0] - min(data[0])

    #Decimate 
    data = decimate(data,decimate_order)

    # get features
    features = fh.get_all_features(data)
    
    return data, features

def preprocess(dfs, lowpass_cutoff, lowpass_order, decimate_order, demean=False, scale=1):
    # decimate_orders = [[10,10], [10,5], [5,5] ,10] #10, 20, 40, 100 Hz
    if decimate_order == '10':
        decimate_order = 10
    elif decimate_order == '20':
        decimate_order = [5,5]
    elif decimate_order == '40':
        decimate_order = [10,5]
    elif decimate_order == '100':
        decimate_order = [10,10]
    
    processed_dfs = []
    feature_arr = []
    if len(dfs) == 1:
        data = dfs[0].to_numpy()
        processed_data, features = process_Ge(data.T, lowpass_cutoff, lowpass_order, decimate_order, demean=False, scale=1)
        processed_dfs.append(processed_data)
        feature_arr.append(features)
    else:
        for df in dfs:
            data = df.to_numpy()
            processed_data, features = process_Ge(data.T, lowpass_cutoff, lowpass_order, decimate_order, demean=False, scale=1)
            processed_dfs.append(processed_data)
            feature_arr.append(features)
    return processed_dfs, feature_arr