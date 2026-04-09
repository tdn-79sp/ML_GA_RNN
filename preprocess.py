import wfdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Very basic data extraction and preprocessing script to get the data into the shape we need for the RNN model
# This was just baseline to get it working but I expanded it a lot in data_extract_resample.py to do the resampling and augmentation for the GA and main training script
# This file is mostly just here for reference and to see the original data extraction logic before I added all the resampling/augmentation stuff

def get_data(record_id):
    # Load the record and annotations
    record = wfdb.rdrecord(f'data/{record_id}')
    annotation = wfdb.rdann(f'data/{record_id}', 'atr')
    
    # Get the signal (Lead II is usually index 0)
    signal = record.p_signal[:, 0]
    
    # Standardize the signal (Mean 0, Std 1)
    signal = StandardScaler().fit_transform(signal.reshape(-1, 1)).flatten()
    
    beats = []
    labels = []
    
    # Mapping MIT-BIH labels to simplified classes
    # N = Normal, V = Ventricular, S = Supraventricular, F = Fusion, Q = Unknown
    label_map = {'N': 0, 'V': 1, 'S': 2, 'F': 3, 'Q': 4}

    for i, (idx, symbol) in enumerate(zip(annotation.sample, annotation.symbol)):
        if symbol in label_map:
            # 180 samples before and after the R-peak (360 total = 1 second at 360Hz)
            if idx > 180 and idx < len(signal) - 180:
                segment = signal[idx-180 : idx+180]
                beats.append(segment)
                labels.append(label_map[symbol])
                
    return np.array(beats), np.array(labels)

# Example usage:
# X, y = get_data('100') 
# X = X.reshape(X.shape[0], X.shape[1], 1) # Reshape for RNN (samples, timesteps, features)