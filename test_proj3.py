#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:31:39 2024

@author: marit scott, sadegh khodabandeloo, ron bryant, michael gallo

This test file does more than test the module proj3.py.  It also serves 
as a notebook to facilitate experiments with the data.  What follows are 
general instructions.  Each cell contains specific instruction for 
setting parameters to control the functions of the proj3 module.

Cells 0 and 1 must be run once per session (or after a reset)
Cell 0) No need to modify, these parameters describe this raw data set.
Cell 1) Loads all subject's data.   (see Cell 7 for a single subject.)

Cell 2) Edit the three parameters (see cell) and run to to epoch the eeg
        data with the desired period, overlap, and censoring criterion.
Cell 3) Edit the three parameters  (see cell) and run to split the epochs
        randomly or in a biased fashion.  If biased 1 or 2 subjects are 
        used for testing and the remainder for training.
Cell 4) Run after changes in Cell 2 or Cell 3 to record them in the 
        eval_parameters dictionary
        
Cells 5), 6), and 7) train the three models on the training sets returned
        from above, and report the accuraccy and generate a confusion
        matrix (can optional be saved to disk)  Also prints sensitivity
        or each class and the false classification rate.
Cell 8) Summarizes the eval_parameters and accuracy from each model.
      
Running the file from the top will execute only Cells 0 to 8. (Assuming the 
if statements of the remaining cells are set to False.) Some of the 
remaining cells generate statistics to compare models and take a lot 
of time to run.

Run Cell 0) before those below to put data_parameters on the workwspace. to pu

Cell 9) Loads and plots a single subject.  Change parameters in cell
Cell 10) Was used to determine the frequency of analog to digital 
        saturation prevalence in the data set.
Cell 11) A runs makes a direct comparison of the three models over an 
        elective number of iterations with given a period, overlap, and 
        splitting criteria
Cell 12a,b,c) "Duplicates" the Acampora et. al. study of comparing logistic 
        regressionto linear discriminant analysis except we compare two 
        neural networks to logistic regression.   The code compares 
        30 period/overlap combinations and enable printing of figures in 
        the same style as in their paper.
Cell 13) Runs paired t-tests on the data from 12.  ???????

    
    
"""

#### Cell 0
import numpy as np
import proj3 as p3
from matplotlib import pyplot as plt
from scipy import stats
    


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

# Three extraction parameters   
period =  6 #seconds
overlap = 0.95   # fractional overlap with prior period
saturation_criterion = 4  # typically use 4 (censoring) or num_times (no censor)
###  To censor epochs containing saturated ADC values set
###  saturation_criterion to 4 (this is a minimal, but reasonable value). 
###  A very large value eliminates any censoring  (e.g. >= num_times=4096)
##################################################################

features, classes, is_censored =      \
        p3.get_feature_vectors(eeg_data, period, overlap, data_parameters,
                               saturation_criterion)

#%% Cell 3 Partition training and testing sets  

# Three splitting parameters
is_random_split = True
proportion_train = 0.8
#if is_random_split test_subjects are ignored,  if False
# Testing set all of the 1 or 2 subjects in the list 
#     and others used for training   
test_subjects = [1,2] # list of 1 or 2 integers from closed set [1,11]
###################################################################


if is_random_split: # Random feature vector assignment to train and test sets    
    training_data, testing_data, training_class, testing_class  =    \
                    p3.split_feature_set(features, classes, is_censored,
                                         proportion_train, data_parameters)
    test_subjects = []
else:  
    
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

# change to a path/filename to save plot
save_to = None
############################

accuracy_4class, _ = p3.train_and_test_NN(training_data, training_class,
                                testing_data, 
                                testing_class, eval_parameters,
                                plot_confusion_table=True, save_to=save_to)

print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' NN 4Class accuracy: {accuracy_4class}')


#%% Cell 6  NN ovr

# change to a path/filename to save plot
save_to = None
##########################################

accuracy_ovr, _ = p3.train_and_test_NN_ovr(training_data, 
                                training_class,
                                testing_data, 
                                testing_class, eval_parameters,
                                plot_confusion_table=True, save_to=save_to)

print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' NN ovr accuracy: {accuracy_ovr}')

#%% Cell 7  LR      (This is Acampora's approach and is an ovr algorythm)

# change to a path/filename to save plot
save_to = None
###########################

accuracy_LR, _ = p3.simple_LR(training_data, training_class,
                           testing_data, 
                           testing_class, eval_parameters,
                           plot_confusion_table=True, save_to=save_to)

print(f'\nAccuracy for Period of {period} with overlap {overlap}')
print(f' LR accuracy: {accuracy_LR}')

#%%  Cell 8
print('\n\n\n\nOverall comparison -- 3 methods ')
print(f'Accuracy for Period of {period} seconds with overlap {overlap}')
print(f'Spliting of data sets is random = {is_random_split}')
print(f'Saturation_criterion = {saturation_criterion}')
print('Accuracy')
print(f'  NN 4 Class: {np.round(accuracy_4class,4)}')
print(f'  NN ovr    : {np.round(accuracy_ovr,4)}')
print(f'  LR        : {np.round(accuracy_LR,4)}')
 

# the following cells test individual funtions
if False:  #leave this as False.  Simply run Cell 9 or Cell 10
#%% Cell 9  Plot and display individual subject
    # define these variables
    is_zoomed = False  # If True plots only a one second interval of data
    subject_num = 7     # Choose from closed set [1,11]
 
    eeg_array = p3.load_subjects(data_parameters,
                        subject=subject_num, # choose from closed set [1,11]
                        zoom=is_zoomed) # or leave out for 'All'
                                        

#%% Cell 10   Tests which of the subjects have saturated the values
              # on the ADC
    for subject in range(11):
        eeg_array = p3.load_subjects(data_parameters, subject= subject+1)
        for freq in range(4):
            print(f'Subject {subject+1}. Saturated low:{np.any(eeg_array[freq,:]==0)}, high:{np.any(eeg_array[freq,:]==1023)}')


#%% Cell 11     Statistical comparison of methods
if False:
    period = 6
    overlap = 0.95
    saturation_criterion = 4
    is_random_split = True
    proportion_train = 0.8
    #if is_random_split test_subjects are ignored,  if False
    # Testing set all of the 1 or 2 subjects in the list 
    #     and others used for training   
    test_subjects = [1,2] # list of 1 or 2 integers from closed set [1,11]
    repetitions = 50
    save_to = 'fig_histo'
    ####################################################################
    
    eval_parameters = {'period' : period,
                       'overlap' : overlap,
                       'saturation_criterion' : saturation_criterion,
                       'is_random_split' : is_random_split,
                       'proportion_train' : proportion_train,
                       'test_subjects' : test_subjects}
    
    p3.compare_models(repetitions, eval_parameters, data_parameters,
                      save_to=save_to)
    
    
                  
#%%  Cell 12a    THIS CELL TAKES AN 30+ MINUTES TO RUN! 5x6x3x10 or more
# Full run of comparisons. 
# Results of runs censored at 4 and  without censoring are saved in
# censiredSet.pkl and fullset0.pkl respectively   plots of each are in
# .png files   fig6*.pkl and fig7*.pkl

if False:
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
    
#%%  Cell 12b load pickle file of the processed data 
    
    #set file_name to an existing valid file produced by above cell
    file_name = 'censoredSet.pkl'
    
    full_set = p3.load_processed_data(file_name) 

    

#%%  Cell 12c Plot a full set of data
                               
    model_idx = 1       # 0,1, or 2 = [ NN4Class, NNovr, LR]
    save_to = None     #None for do not save or give path/file name
    
    p3.plot_cross_val(full_set, model_idx, save_to)



########################################################################
######   BELOW HERE IS DATA ASSESSEMNT   ####################
#############################################################

#%% Cell 13 Run paired t-test on selected epoch sets  
## I didn't write a function to do this because that would lead to a 
## doing a bunch of t-tests and risking a bunch of false discoveries.
##  These are the only ones we felt the need to do.  Also quick and dirty 
##  histograms visualize the data.

if False: 
    # load censored data -- an epoch is censored if >= 4 consecutive 
                                # saturated time points occur in the epoch
    file_name = 'censoredSet.pkl'
    full_set_censored = p3.load_processed_data(file_name)
    
    # load uncensored data
    file_name = 'fullset0.pkl'
    full_set_uncensored = p3.load_processed_data(file_name)
    
    # get accuracy arrays
    censored = full_set_censored['full_accuracy_array']
    uncensored = full_set_uncensored['full_accuracy_array']

    # Compare NNovr to LR for uncensored  (indices to full_set data arrays
          # are period_idx  x  overlap_idx  x  model_idx  x  iteration_idx)
    mean_NNovr = np.round(np.mean(uncensored[4,5,1,:])*100,1)
    std_NNovr = np.round(np.std(uncensored[4,5,1,:])*100,1)
    mean_LR = np.round(np.mean(uncensored[4,5,2,:])*100,1)
    std_LR = np.round(np.std(uncensored[4,5,2,:])*100,1)
    t_test, p_val = stats.ttest_rel(uncensored[4,5,1,:],uncensored[4, 5, 2,:])
 
    print('\n\nNNovr versus LR (uncensored, 6 second periods with overlap of 95%)')
    print(f'  NNovr accuracy(%):  {mean_NNovr} +/- {std_NNovr} (mean +/- sd, n = 10')
    print(f'  LR accuracy(%):  {mean_LR} +/- {std_LR} (mean +/- sd, n = 10)')
    print(f'  Paired t-statistic = {np.round(t_test,2)},  p-value = {np.round(p_val,4)}\n')
    plt.figure(num= 1, clear=True)
    plt.hist(uncensored[4,5,1,:], alpha = 0.5)
    plt.hist(uncensored[4,5,2,:], alpha = 0.5)


    # Compare NN4Class to LR for uncensored  (indices to full_set data arrays
          # are period_idx  x  overlap_idx  x  model_idx  x  iteration_idx)
    mean_NN4Class = np.round(np.mean(uncensored[4,5,0,:])*100,1)
    std_NN4Class = np.round(np.std(uncensored[4,5,0,:])*100,1)
    t_test, p_val = stats.ttest_rel(uncensored[4,5,0,:],uncensored[4, 5, 2,:])
 
    print('\n\nNN 4Class versus LR (uncensored, 6 second periods with overlap of 95%)')
    print(f'  NN 4Class accuracy(%):  {mean_NN4Class} +/- {std_NN4Class} (mean +/- sd, n = 10')
    print(f'  LR accuracy(%):  {mean_LR} +/- {std_LR} (mean +/- sd, n = 10)')
    print(f'  Paired t-statistic = {np.round(t_test,2)},  p-value = {np.round(p_val,4)}\n')
    plt.figure(num= 2, clear=True)
    plt.hist(uncensored[4,5,0,:], alpha = 0.5)
    plt.hist(uncensored[4,5,2,:], alpha = 0.5)
    
    
    # Compare NNovr uncensored to NNovr censored  (indices to full_set data arrays
          # are period_idx  x  overlap_idx  x  model_idx  x  iteration_idx)
    mean_NNovr_cen = np.round(np.mean(censored[4,5,1,:])*100,1)
    std_NNovr_cen = np.round(np.std(censored[4,5,1,:])*100,1)
    t_test, p_val = stats.ttest_rel(uncensored[4,5,1,:],censored[4, 5, 1,:])
 
    print('\n\nNNovr (uncensored) versus NNovr (censored)  ( 6 second periods with overlap of 95%)')
    print(f'  NNovr (uncensored) accuracy(%):  {mean_NNovr} +/- {std_NNovr} (mean +/- sd, n = 10')
    print(f'  NNovr (censored) accuracy(%):  {mean_NNovr_cen} +/- {std_NNovr_cen} (mean +/- sd, n = 10)')
    print(f'  Paired t-statistic = {np.round(t_test,2)},  p-value = {np.round(p_val,4)}\n')
    plt.figure(num= 3, clear=True)
    plt.hist(uncensored[4,5,1,:], alpha = 0.5)
    plt.hist(censored[4,5,1,:], alpha = 0.5)

    
    plt.figure(num= 4, clear=True)
    overlaps = full_set_uncensored['periods']
    max_NNovr_uncen = np.max(np.max(uncensored[:,:,1,:],axis = -1), axis = 1)
    max_LR_uncen = np.max(np.max(uncensored[:,:,2,:],axis = -1), axis = 1)
    max_NNovr_cen = np.max(np.max(censored[:,:,1,:],axis = -1), axis = 1)
    plt.figure(num=4, clear = True)
    plt.plot(overlaps,max_NNovr_cen, '.-', label = 'NNovr censored')
    plt.plot(overlaps,max_NNovr_uncen, '.-',  label = 'NNovr uncensored')
    plt.plot(overlaps,max_LR_uncen, '.-', label = 'LR uncensored')
    plt.xlim(1.5, 6.5)
    plt.xlabel('Epoch period (Seconds)')
    plt.ylim(0.5,1)
    plt.ylabel('Accuracy (proportion)')
    plt.legend(loc = 4)
    plt.title('Testing Max Scores (95% overlap)')
    plt.grid()

    plt.show()
    plt.savefig('fig8')

#%%   Not used
    if False:
        plt.figure(num= 5, clear=True)
        overlaps = full_set_uncensored['periods']
        mean_NNovr_uncen = np.max(np.mean(uncensored[:,:,1,:],axis = -1), axis = 1)
        mean_LR_uncen = np.max(np.mean(uncensored[:,:,2,:],axis = -1), axis = 1)
        mean_NNovr_cen = np.max(np.mean(censored[:,:,1,:],axis = -1), axis = 1)
        plt.figure(num=4, clear = True)
        plt.plot(overlaps,mean_NNovr_cen, '.-', label = 'NNovr censored')
        plt.plot(overlaps,mean_NNovr_uncen, '.-',  label = 'NNovr uncensored')
        plt.plot(overlaps,mean_LR_uncen, '.-', label = 'LR uncensored')
        plt.xlim(1.5, 6.5)
        plt.xlabel('Epoch period (Seconds)')
        plt.ylim(0.5,1)
        plt.ylabel('Accuracy (proportion)')
        plt.legend(loc = 4)
        plt.title('Testing Max Scores')
        plt.grid()
        
        plt.show()
        
#%% Cell  Constructing a NNclass to perform back prop
            # works well (in same ball park as the other 3 models), but
            # haven't had chance to program in any optimizers.
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
        
