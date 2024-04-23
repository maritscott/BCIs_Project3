#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:31:39 2024

@author: marit scott, sadegh khodabandeloo, ron bryant

TO DO:
    Add confusion table, and figures of merit
    Add my NN Class
    load_subject to return all if requested.
    allow splitting of one or two subjects for testing  NEEDS COMPLETION
    label time graph
"""
import numpy as np
import proj3 as p3
#from matplotlib import pyplot as plt
#import pandas as pd

# dataset parameters  defined for source data set
fs = 256             # sampling frequency
num_subjects = 11    # data files indexed from 1 to 11
num_frequencies = 4  #  
frequencies = [8.57, 10, 12, 15]  # Hz
num_times = 4096     # 16 seconds durations
data_parameters = {'fs' : fs, 'num_subjects' : num_subjects,
                   'num_frequencies' : num_frequencies,
                   'frequencies' : frequencies,
                   'num_times' : num_times }

#%%   EEG of single subject plotted
eeg_array = p3.load_subjects(data_parameters,
                             subject=3, zoom=False) # choose from closed set [1,11]
                                          # or leave out for 'All'

#%%   Load All Data

eeg_data = p3.load_subjects(data_parameters)
    

#%%  Extract features

# extraction parameters   
period = 6   #seconds
overlap = 0.95   # fractional overlap with prior period

features, classes = p3.get_feature_vectors(eeg_data, period, 
                                           overlap, data_parameters)



#%%  Randomize and partition training and testing sets

if True: # Random feature vectpr assignment tot train and test sets
    proportion_train = 0.8
    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.split_feature_set(features, classes,
                                         proportion_train, data_parameters)

else:  # Testing set all of 1 or 2 subjects and others used for training   
    test_subjects = [1,2] # list of 1 or 2 integers from closed set [1,11]
    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.biased_split_feature_set(features, classes,
                                                data_parameters, 
                                                test_subjects)


#%%  LR

freq_to_test = 3  # valid choices : 0 to 3 (or omit in call for 4-class)
accuracy_4class, _ = p3.train_and_test_LR(training_data, training_class,
                                testing_data, 
                                testing_class)#, single_freq=freq_to_test)
print(f'Accuracy for Period of {period} with overlap {overlap}')
print(f' is {accuracy_4class}')


#%%  LR_ovr

accuracy_ovr = p3.train_and_test_LR_ovr(training_data, 
                                    training_class,
                                    testing_data, 
                                    testing_class)
print(f'ovr accuracy = {accuracy_ovr}')

#%% 
print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' LR accuracy is {np.round(accuracy_4class,5)}')
print(f' My ovr accuracy is {np.round(accuracy_ovr,5)}')



