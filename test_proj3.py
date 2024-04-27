#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:31:39 2024

@author: marit scott, sadegh khodabandeloo, ron bryant, michael gallo

TO DO:
    Add  figures of merit
    Add my NN Class
    
    
"""

#### Cell 0
import numpy as np
import proj3 as p3
from matplotlib import pyplot as plt

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

#%% Cell 1  Load All Data

eeg_data = p3.load_subjects(data_parameters)
    

#%% Cell 2 Extract features

# extraction parameters   
period =  4 #seconds
overlap = 0.95   # fractional overlap with prior period
saturation_criterion = num_times
###  To censor epochs containing saturated ADC values set
###  saturation_criterion to 4 (this is a minimal, but reasonable value). 
###  A very large value eliminates any censoring  (e.g. >= num_times)

features, classes, is_censored =      \
        p3.get_feature_vectors(eeg_data, period, overlap, data_parameters,
                               saturation_criterion)

#%% Cell 3 Partition training and testing sets  

is_random_split = True

if is_random_split: # Random feature vector assignment to train and test sets
    proportion_train = 0.8
    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.split_feature_set(features, classes, is_censored,
                                         proportion_train, data_parameters)
    test_subjects = []
else:  # Testing set all of 1 or 2 subjects and others used for training   
    test_subjects = [1,2] # list of 1 or 2 integers from closed set [1,11]
    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.biased_split_feature_set(features, classes, 
                                                is_censored,
                                                data_parameters, 
                                                test_subjects)
    n_subs = data_parameters['num_subjects']
    proportion_train = (n_subs - len(test_subjects))/n_subs
#%% Cell 4  Record evaluation parameters to pass to function report()
eval_parameters = {'period' : period,
                   'overlap' : overlap,
                   'saturation_criterion' : saturation_criterion,
                   'is_random_split' : is_random_split,
                   'proportion_train' : proportion_train,
                   'test_subjects' : test_subjects}

#%% Cell 5  NN 4Class


accuracy_4class, _ = p3.train_and_test_NN(training_data, training_class,
                                testing_data, 
                                testing_class, eval_parameters,
                                plot_confusion_table=True)

print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' NN 4Class accuracy: {accuracy_4class}')


#%% Cell 6  LR_ovr

accuracy_ovr = p3.train_and_test_NN_ovr(training_data, 
                                    training_class,
                                    testing_data, 
                                    testing_class, eval_parameters,
                                    plot_confusion_table=True)

print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' NN ovr accuracy: {accuracy_ovr}')

#%% Cell 7  LR ovr

accuracy_LR = p3.simple_LR(training_data, training_class,
                                    testing_data, 
                                    testing_class, eval_parameters,
                                    plot_confusion_table=True)

print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' LR ovr accuracy: {accuracy_LR}')

#%%  Cell 8
print('\n\n\n\nOverall comparison -- 3 methods ')
print(f'Accuracy for Period of {period} seconds with overlap {overlap}')
print(f'Spliting of data sets is random = {is_random_split}')
print(f'Saturation_criterion = {saturation_criterion}')
print('Accuracy')
print(f'  NN 4 Class: {np.round(accuracy_4class,4)}')
print(f'  NN ovr    : {np.round(accuracy_ovr,4)}')
print(f'  LR ovr    : {np.round(accuracy_LR,4)}')
 

# the following cells test individual funtions
if False:
#%% Cell 9  Plot and display individual subject
    eeg_array = p3.load_subjects(data_parameters,
                             subject=7, # choose from closed set [1,11]
                             zoom=False) # or leave out for 'All'
                                        

#%% Cell 10   Tests for which of the subjects have saturated the values
              # on the ADC
    for subject in range(11):
        eeg_array = p3.load_subjects(data_parameters, subject= subject+1)
        for freq in range(4):
            print(f'Subject {subject+1}. Saturated low:{np.any(eeg_array[freq,:]==0)}, high:{np.any(eeg_array[freq,:]==1023)}')


#%% Cell 11  Constructing a NNclass to perform back prop

if False:   # UNDER CONSTRUCRTION
    from NNclass import *
    
    xtr = training_data.T
    xte = testing_data.T 
    feature_len = training_data.shape[1]
    trlen = training_class.shape[0]
    telen = testing_class.shape[0]
    # Class defined by one-hot coding
    ctr = np.zeros((4,trlen))
    cte = np.zeros((4,telen))
    for i in range(trlen):
        ctr[training_class[i].astype(int), i] = 1
    for i in range(telen):
        cte[testing_class[i].astype(int), i] = 1
    # intstiate and run NN    
    nn = NeuralNetwork([feature_len,10,4],learning_rate=0.1)
    nn.train(xtr,ctr,200)#[:,trpoints],y[:,trpoints],epochs = 1000)
    
    print('\n\n\nHand made 2-layer 4-class NN with back propogation')
    print('(in process of development)')
    print(f' Train score = {nn.score2(xtr,training_class)}%')
    print(f' Test score = {nn.score2(xte,testing_class)}%')

#%% Cell 12     Statistical comparison of methods
# This cell runs each of 3 networks through multiple repetions, 
# resplitting training and testing epochs data each time and compares
#  the accuracy of the three training methods (as per Acampora et. al.)

############# To Run this cell ##############
#    1) Go to cells 2 and 3  to select  epoching and splitting paramenters
#    2) Run cells 0,1,2,3,4
#    3) Define repetitions below
#    4) Set first line below to True
#    5) Run this cell.

if True:
    repetitions = 20   # This was used Acampora et. al.

    #####################################################################
    # Prepare accuracy_array
    accuracy_array = np.zeros([3, repetitions])

    for rep_index in range(repetitions):

        accuracy_array[0, rep_index], _ =    \
               p3.train_and_test_NN(training_data, training_class,
                                    testing_data, 
                                    testing_class, eval_parameters)

        accuracy_array[1, rep_index] =       \
               p3.train_and_test_NN_ovr(training_data, 
                                        training_class,
                                        testing_data, 
                                        testing_class, eval_parameters)

        accuracy_array[2, rep_index] =        \
               p3.simple_LR(training_data, training_class,
                                        testing_data, 
                                        testing_class, eval_parameters)
        # resplit       
        if is_random_split: # Random feature vector assignment to train and test sets
            training_data, testing_data, training_class, testing_class  =    \
                        p3.split_feature_set(features, classes, is_censored,
                                             proportion_train, data_parameters)
        else:  # Testing set all of 1 or 2 subjects and others used for training   
            training_data, testing_data, training_class, testing_class  =    \
                        p3.biased_split_feature_set(features, classes, 
                                                    is_censored,
                                                    data_parameters,
                                                    test_subjects) 
        print(f'Finished repetition {rep_index}  . . .')
    means = np.round(np.mean(accuracy_array, axis = -1),4)
    sds = np.round(np.std(accuracy_array, axis = -1),4)          
    print(f'\n\n\nRepetitions = {repetitions}')
    print(f'Accuracy for Period of {period} seconds with overlap {overlap}')
    print(f'Spliting of data sets is random = {is_random_split}')
    print(f'Training proportion is {proportion_train}')
    print(f'Saturation_criterion = {saturation_criterion}')
    print('Accuracy')
    print(f'   NN 4Class: {means[0]} +/- {sds[0]}' )
    print(f'   NN ovr   : {means[1]} +/- {sds[1]}' )
    print(f'   LR ovr   : {means[2]} +/- {sds[1]}' )
              
#%%  Full run of comparisons. THIS CELL TAKES AN HOUR TO RUN! 5x6x3x10
# Results of run with (censored at 4) without censoring are saved in
# censiredSet.pkl and fullset).pkl respectively   plots of each are in
# .png files   fig6*.pkl and fig7*.pkl

if True:
    # full set of training options per Acampora et. al.
    periods = [2,3,4,5,6]
    overlaps = [0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
    models = ['NN 4Class', 'NN ovr', 'LR']
    iterations = 10
    
    # Set these two   save_to  and eval_parameters
    #data_parameters defined in Cell 0 for the data set
    save_to = 'test'     # data is saved to the this file and returned
    # leave these alone except possibly for 
    # saturation_citerion   and   proportion_train.
    eval_parameters = {'period' : periods[0],
                       'overlap' : overlaps[0],
                       'models' : models,
                       'saturation_criterion' : 4096,   # use 4 or 4096
                       'is_random_split' : True,
                       'proportion_train' : 0.8,
                       'test_subjects' : []}
    
    
    full_set = p3.run_set(data_parameters, eval_parameters, save_to,
                          periods=periods, overlaps=overlaps, 
                          models=models, iterations=iterations )
    
#%%   load pickle file of the processed data 
    
    #set file_name to an existing valid file produced by above cell
    file_name = 'fullset0.pkl'
    
    full_set = p3.load_processed_data(file_name) 

    

    #%%   
                               
model_idx = 2
p3.plot_cross_val(full_set, model_idx, save_to='fig7LR')


