#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:31:39 2024

@author: marit scott, sadegh khodabandeloo, ron bryant, michael gallo

TO DO:
    Add confusion table, and figures of merit
    Add my NN Class
    load_subject to return all if requested.
    allow splitting of one or two subjects for testing  NEEDS COMPLETION
    label time graph
"""
import numpy as np
import proj3 as p3
from matplotlib import pyplot as plt
#import pandas as pd

# dataset parameters  defined for source data set
fs = 256             # sampling frequency Hz
num_subjects = 11    # data files indexed from 1 to 11
num_frequencies = 4  #  
frequencies = [8.57, 10, 12, 15]  # Hz
num_times = 4096     # 16 seconds durations
data_parameters = {'fs' : fs, 'num_subjects' : num_subjects,
                   'num_frequencies' : num_frequencies,
                   'frequencies' : frequencies,
                   'num_times' : num_times }

#%%   Load All Data

eeg_data = p3.load_subjects(data_parameters)
    

#%%  Extract features

# extraction parameters   
period = 6   #seconds
overlap = 0.95   # fractional overlap with prior period
saturation_criterion = 4000
###  To censor epochs containing saturated ADC values set
###  saturation_criterion to 4 (this is a minimal, but reasonable value). 
###  A very large value eliminates any censoring  (e.g. 4000)



features, classes, is_censored =      \
        p3.get_feature_vectors(eeg_data, period, overlap, data_parameters,
                               saturation_criterion)



#%%  Partition training and testing sets  

random_split = True

if random_split: # Random feature vectpr assignment to train and test sets
    proportion_train = 0.85
    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.split_feature_set(features, classes, is_censored,
                                         proportion_train, data_parameters)

else:  # Testing set all of 1 or 2 subjects and others used for training   
    test_subjects = [1,2] # list of 1 or 2 integers from closed set [1,11]
    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.biased_split_feature_set(features, classes, 
                                                is_censored,
                                                data_parameters, 
                                                test_subjects)


#%%  NN 


accuracy_4class, _ = p3.train_and_test_NN(training_data, training_class,
                                testing_data, 
                                testing_class)

print(f'Accuracy for Period of {period} with overlap {overlap}')
print(f' NN (4 Class); {accuracy_4class}')


#%%  LR_ovr

accuracy_ovr = p3.train_and_test_NN_ovr(training_data, 
                                    training_class,
                                    testing_data, 
                                    testing_class)

print(f'Accuracy for Period of {period} with overlap {overlap}')
print(f'ovr accuracy = {accuracy_ovr}')

#%%  Simple LR

accuracy_LR = p3.simple_LR(training_data, training_class,
                                    testing_data, 
                                    testing_class)

#%% 
print(f'\nAccuracy for Period of {period} seconds with overlap {overlap}')
print(f'Spliting of data sets is random = {random_split}')
print(f'Saturation_criterion = {saturation_criterion}')
print(f'  NN (4 Class): {np.round(accuracy_4class,4)}')
print(f'  NN (ovr)    : {np.round(accuracy_ovr,4)}')
print(f'  LR (ovr)    : {np.round(accuracy_LR,4)}')


# the following cells test individual funtions
if False:
#%%   Plot and display individual subject
    eeg_array = p3.load_subjects(data_parameters,
                             subject=7, # choose from closed set [1,11]
                             zoom=False) # or leave out for 'All'
                                        

#%%  Tests for which of the subjects have saturated the values on the ADC
    for subject in range(11):
        eeg_array = p3.load_subjects(data_parameters, subject= subject+1)
        for freq in range(4):
            print(f'Subject {subject+1}. Saturated low:{np.any(eeg_array[freq,:]==0)}, high:{np.any(eeg_array[freq,:]==1023)}')
