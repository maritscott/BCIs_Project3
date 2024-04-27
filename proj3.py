#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:08:45 2024

@author: marit scott, sadegh khodabandeloo, ron bryant, michael gallo
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
import pickle


#from tensorflow.keras.layers import Dense, Flatten, Conv2D
#from tensorflow.keras import Model


def load_subjects(params, subject='All', zoom=False):
    '''
    Loads EED data from disk.  All data is loaded by default.  If a 
    single subject is designated the EEG is plotted.     

    Parameters
    ----------
    params : Dictionary    
        DESCRIPTION.  Contains basic parameters describing the data files.
                    See the test file for specifics for this data set.
    subject : Optional integer
        DESCRIPTION. The default is 'All'.  of an integer from the closed
                        set [0,11] is entered, only that individuals data
                        is loaded and the EEG data is plotted.
    zoom : boolean    default False
        DESCRIPTION.  If set to True and a single subject selected the
                    eeg is displayed for just 1 sec of data (vs 16)

    Returns
    -------
    eeg_array : 2d array of float    num_eegs x num_times
        DESCRIPTION.  EEG from the selected subject (or all subjects)
                    at 4 frequencies are returned. Each subject's EEGs 
                    are in adjacent rows. 
                    The EEG values are integers in the closed interval
                    [0, 1023].  The sampling is 4096 time point at 
                    256 Hz = 16 seconds. 
    '''
    
    def load_one(subject_number, num_freqs):
    
        '''
        The file for the subject is read and the eeg at each stimulus
        frequency is returned.

        Parameters
        ----------
        subject_number : integer from the closed set [0, 11]
            DESCRIPTION.  Data for this subject is loaded
        num_freqs : integer 
            DESCRIPTION. number stimulus frequencies in file

        Returns
        -------
        eeg : 2d array of float    num_freqs x num_times
            DESCRIPTION. A subject's EEG from num_freqs stimulation 
                        frequencies. The EEG values are integers in the 
                        closed interval [0, 1023].  The sampling is 
                        4096 time points at 256 Hz = 16 seconds. 
        '''
        
        df = pd.read_csv(f'subject{subject_number}.csv') 
                                                # not actually a CSV
        len_eeg = df.shape[0]
        #initialize
        eeg = np.zeros((num_freqs, len_eeg))
        # read line-by-line and populate eeg
        for df_index in range(len_eeg):
            df_line = df.iloc[df_index].values  # a ';' separated string 
            str_list = df_line[0].split(';')
            eeg[:,df_index] = [int(x) for x in str_list]
        return eeg



    num_freqs = params['num_frequencies']
    len_eeg = params['num_times']

    if subject == 'All':
        n_subjects = params['num_subjects']
        #initialize
        eeg_array = np.zeros((n_subjects * num_freqs, len_eeg))
        # populate with data from each subject
        for sub_num in range(n_subjects):
            eeg_array[(sub_num*num_freqs):((sub_num+1)*num_freqs) ,:] =   \
                            load_one(sub_num+1, num_freqs)
                            
                            
    else:  # load indicated subject and plot EEG for each stimulus
        eeg_array = load_one(subject, num_freqs)
        # display data
        freqs = params['frequencies']
        time = np.arange(0,len_eeg)/params['fs']
        plt.figure(1, figsize = (8,8), clear = True)
        
        for i in range(4):
            plt.subplot(4,1,i+1)
            plt.plot(time, eeg_array[i, :])
            plt.ylabel('Voltage')
            plt.title(f'Stimulus Frequency {freqs[i]} Hz')
            if zoom:
                plt.xlim(7,8)
        plt.xlabel('Time (seconds)')
        
        plt.suptitle(f'EEG by Stimulus Frequencies, Subject {subject} (voltage units unknown)')
        plt.tight_layout()
        plt.show()    
        
    return eeg_array    


def is_saturated(epoch, criterion):
    '''
    Helper function for epoch_data() determines if an epoch has saturated
    it's ADC for more than  criterion  consequtive time points

    Parameters
    ----------
    epoch : TYPE     1d array of integers 
        DESCRIPTION.   EEG values in close interval [0, 1023] from the ADC
    criterion : TYPE  interger
        DESCRIPTION.   if  criterion 0's or 1023's apear sequentially in
                the epoch the function returns True  (else False)

    Returns
    -------
    bool    
        DESCRIPTION.  see criterion above

    '''
    one_less = criterion - 1
    #detects the indices wher there are cosecutive 0's or 1023's
    inds_low = np.where(epoch == 0)[0]
    inds_high = np.where(epoch == 1023)[0]
    # nuber of each
    n_inds_low = inds_low.shape[0]
    n_inds_high = inds_high.shape[0]
    # doesn't exceed criterion return False
    if (n_inds_low < criterion) and  (n_inds_high < criterion):
        return False
    #might exceed criterion   Look at 0's
    if n_inds_low > one_less:
        for idx in range(n_inds_low-one_less):
            if (inds_low[idx]) == (inds_low[idx+one_less] - one_less):
                return True
    # then look at 1023's
    if n_inds_high >one_less:
        for idx in range(n_inds_high - one_less):
            if (inds_high[idx]) == (inds_high[idx + one_less] - one_less):
                return True
    #couldn't find enough consecutive saturations
    return False
  

def epoch_data(eeg_data, time_period, overlap, parameters, 
               saturation_criterion):
    '''
    Helper function for ger_feature_vectors()
    Return epochs of data of length time_period statrting from 
    the begining of eeg_data  with overlapping of time period of 
    proportion overlap.

    Parameters
    ----------
    eeg_data : 2d array of float    num_freqs x num_times
        DESCRIPTION.  an EEG from num_freqs stimulation frequencies 
                    The EEG values are in arbitray units since an
                    unknown amplification occurs during acquisition.
                    The sampling is 4096 point at 256 Hz = 16 seconds.
    time_period : float  from the open interval (0, 16)
        DESCRIPTION.  Duration of each epoch
    overlap : float    from open interval (0, 1)
        DESCRIPTION.  amount intervals may overlap.  With a large 
                    overlap many epochs can be obtained from the 
                    16 second EEG even with a relatively large time_period
    parameters : Dictionary    
        DESCRIPTION.  Contains basic parameters describing the data files.
                    See the test file for specifics for this data set. 
                    Includes fs (sampling freqeuncy)
    saturation_criterion : integer 
        DESCRIPTION.  Number of consecutive eeg values at maximum (1023)
                        or minimum (0) to qualify for censoring an epoch.
                        A large value (4000) eliminates censoring.   4 is
                        probably a minimal meaningful value.
        
    Returns
    -------
    epochs : 2d array of float of size 
                (n_subjects * n_frequencies * n_epochs_per_eeg) x n_times
                n_times = time_period * sampling frequency
        DESCRIPTION.   The EEG values for the epochs. The EEG voltages are 
                        in arbitray units since an unknown amplification 
                        occurs during acquisition.
  
    epoch_classes : 1d array integers from the closed set [0,3]
        DESCRIPTION.   These are the 4 classes that represent the 
                      stimulus frequencies for each subject. In the epochs 
                      the classes are in adjacent rows and are in order of 
                      ascending stimulus frequency and each subject has 
                      n_epochs_per_eeg in each class (all adjacent rows).
    is_censored : 1d array of booleans 
        DESCRIPITION. Indicaates which of the epochs meet censoring 
                    criterion.
    '''
    
    
    # data_parameters
    fs = parameters['fs']
    num_freqs = parameters['num_frequencies']
    num_indices = eeg_data.shape[-1] 
    index_period = time_period * fs
    overlap_offset = int(index_period * overlap)

    # determine lists of epoch start and end times  
    start_times = [0]
    end_times = [index_period]
    while (end_times[-1] - overlap_offset + index_period) <= \
                                        num_indices:
        start_times.append(end_times[-1] - overlap_offset)
        end_times.append(start_times[-1] + index_period)
    
    if False:  # visualize segemntation of data to form epochs
        print(start_times)
        print(end_times)
    
    # initialize epochs
    num_epochs_per_eeg = len(start_times)
    num_epochs = eeg_data.shape[0] * num_epochs_per_eeg
    epochs = np.zeros((num_epochs, index_period))
    epoch_classes = np.zeros((num_epochs))
    
    # populate epochs
    for eeg_index in range(eeg_data.shape[0]):
        for epoch_per_eeg_index in range(num_epochs_per_eeg):
            epoch_index = eeg_index * num_epochs_per_eeg       \
                                + epoch_per_eeg_index 
            epochs[epoch_index,:] =  \
                    eeg_data[eeg_index,                        \
                    start_times[epoch_per_eeg_index]:   \
                                            end_times[epoch_per_eeg_index]]
            epoch_classes[epoch_index] = eeg_index % num_freqs
    
    is_epoch_censored = np.zeros(num_epochs, dtype=bool)
    # determine if the epochs needs to be censored.
    for epoch_index in range(num_epochs):
        is_epoch_censored[epoch_index] = is_saturated(epochs[epoch_index],
                                                      saturation_criterion)
 
    return  epochs, epoch_classes, is_epoch_censored


def get_feature_vectors(eeg_data, period, overlap, params,
                        saturation_criterion=4):
    '''
    Epochs the eeg_data based on the period and overlap and generates 
    and returns feature vectors and classes for each epoch.

    Parameters
    ----------
    eeg_data : eeg : 2d array of float 
                    (num_subjects * num_stim_freqs) x num_times
        DESCRIPTION.  for each s.ubject an EEG from num_freqs stimulation 
                    frequencies The EEG values are in arbitray units since
                    an unknown amplification occurs during acquisition.
                    The sampling is 4096 point at 256 Hz = 16 seconds.
    period : float  from the open interval (0, 16)
        DESCRIPTION.  Duration of each epoch
    overlap : float    from open interval (0, 1)
        DESCRIPTION.  amount intervals may overlap.  With a large 
                    overlap many epochs can be obtained from the 
                    16 second EEG even with a relatively large time_period
    params : Dictionary    
        DESCRIPTION.  Contains basic parameters describing the data files.
                    See the test file for specifics for this data set. 
                    Includes fs (sampling freqeuncy)
    saturation_criterion : interger (optional with default of 4)
        DESCRIPTION.  Number of consecutive eeg values at maximum (1023)
                        or minimum (0) to qualify for censoring an epoch.
                        A large value (4000) eliminates censoring.   4 is
                        probably a minimal meaningful value.
                        
    Returns
    -------
    feature_vectors : 2d array of float   
                        (num_subjects * num_stim_freqs) x n_freq_amplitudes
        DESCRIPTION.  The FFT over the interval 4 to 32 Hz.  The lenght of
                    n_freq_amplitudes will vary with the length of the 
                    period since frequeny step = 1/period
    vector_classes : 1d array integers from the closed set [0,3]
        DESCRIPTION.   These are the 4 ccorresponding classes for 
                      each feature vectors. In the feature_vectors 
                      the classes are in adjacent rows and are in order of 
                      ascending stimulus frequency and each subject has 
                      n_epochs_per_eeg in each class (all adjacent rows).
    is_censored : 1d array of booleans 
        DESCRIPITION. Indicaates which of the epochs meet censoring 
                    criterion.
    '''
    
    epochs, vector_classes, is_censored    \
                 = epoch_data(eeg_data, period, overlap,   \
                                        params, saturation_criterion)

    # fft of epochs
    ffts = abs(np.fft.rfft(epochs,axis=-1))
    
    # select range of frequencies for feature vector
    df = 1/period  # frequency resolution
    freq_lower = 4         # 4 to 32 Hz encompasses all stimuli
    freq_higher = 32 
    index_lower = int(round(freq_lower/df))
    index_higher = int(round(freq_higher/df))
    
    features  = ffts[:,index_lower:index_higher]
    #noramlize features
    feature_vectors = (features - features.mean(axis=-1, keepdims=True) )   \
                    / features.std(axis=-1,keepdims=True)
    
    return feature_vectors, vector_classes, is_censored  




def split_feature_set(features, classes, is_censored,
                      proportion_train, parameters):
    '''
    Randomly splits the feature vectors into training sets and testing sets
    after censoring based on is_censored
    
    Parameters
    ----------
    features : 2d array of float   
                        (num_subjects * num_stim_freqs) x n_freq_amplitudes
        DESCRIPTION.  The FFT over the interval 4 to 32 Hz.  The length of
                    n_freq_amplitudes will vary with the length of the 
                    period since frequeny step = 1/period
    classes : 1d array  integers from the closed set [0,3]
        DESCRIPTION.   These are the 4 corresponding classes that represent 
                      for each feature vectors In the feature vectors 
                      the classes are in adjacent rows and are in order of 
                      ascending stimulus frequency and each subject has 
                      n_epochs_per_eeg in each class (all adjacent rows).    
    is_censored : 1d array of booleans 
        DESCRIPITION. Indicaates which of the epochs meet censoring 
                    criterion.
    proportion_train : float  from open inteval (0,1)
        DESCRIPTION.  Proportion of data to put in traing set.  Remainder is 
                    put in testing set.
    parameters : Dictionary    
        DESCRIPTION.  Contains basic parameters describing the data files.
                    See the test file for specifics for this data set. 
                    Includes fs (sampling freqeuncy)

    Returns
    -------
    training_set : 2d array of float  N_feature vectors x n_freq_amplitudes
        DESCRIPTION.  A random selection of proportion_train of the features.
    testing_set : 2d array of float  M_feature vectors x n_freq_amplitudes
        DESCRIPTION.  The remaining feature vectors.
    training_class : 1d arrary  from closed set [0,3]
        DESCRIPTION.  Classes of the  training set.
    testing_class : 1d arrary  from closed set [0,3]
        DESCRIPTION.  Classes of the testing set
    '''
    
    #removed censored
    features = features[~is_censored]
    classes = classes[~is_censored]
    
    #generate a random permutation of the uncensored feature vectors
    set_indices = np.arange(features.shape[0])
    permuted_indices = np.random.permutation(set_indices)
    
    # choose the proportion of training examples
    num_train = int(features.shape[0] * proportion_train)
    # first num_train vectors are training, rest are testing
    training_set = features[ permuted_indices[:num_train] ]
    testing_set = features[ permuted_indices[num_train:] ]
    # similarly for classes
    training_class = classes[ permuted_indices[:num_train] ]
    testing_class = classes[ permuted_indices[num_train:] ]
    
    return training_set, testing_set, training_class, testing_class 



def biased_split_feature_set(features, classes, is_censored, 
                             parameters, test_list):
    '''
    Returns a biased testing set of the all epochs from the 1 or 2 
    subjects in the test_list (after censoring based on is_censored)

    Parameters
    ----------
    features : 2d array of float   
                        (num_subjects * num_stim_freqs) x n_freq_amplitudes
        DESCRIPTION.  The FFT over the interval 4 to 32 Hz.  The length of
                    n_freq_amplitudes will vary with the length of the 
                    period since frequeny step = 1/period
    classes : 1d array  integers from the closed set [0,3]
        DESCRIPTION.   These are the 4 corresponding classes that represent 
                      for each feature vectors In the feature vectors 
                      the classes are in adjacent rows and are in order of 
                      ascending stimulus frequency and each subject has 
                      n_epochs_per_eeg in each class (all adjacent rows).    
    is_censored : 1d array of booleans 
        DESCRIPITION. Indicaates which of the epochs meet censoring 
                    criterion.
    parameters : Dictionary    
        DESCRIPTION.  Contains basic parameters describing the data files.
                    See the test file for specifics for this data set. 
                    Includes fs (sampling freqeuncy)
    test_list : list of integers of length 1 or 2 with integers from
                closed list [1,11]
        DESCRIPTION.  List of one or two subjects to put in testing set.
                        The remaining subjuects are in the training set.
    Returns
    -------
    training_set : 2d array of float  N_feature vectors x n_freq_amplitudes
        DESCRIPTION.  A biased selection of the features.
    testing_set : 2d array of float  M_feature vectors x n_freq_amplitudes
        DESCRIPTION.  The remaining feature vectors.
    training_class : 1d arrary  from closed set [0,3]
        DESCRIPTION.  Classes of the  training set.
    testing_class : 1d arrary  from closed set [0,3]
        DESCRIPTION.  Classes of the testing set
    '''
    
    
    num_feature_vectors = features.shape[0]
    features_per_subject = int(num_feature_vectors / parameters['num_subjects'])
 
    #These few lines take advantage of the ordering of the feature vectors
    #in the array to select the subjects in test_list to be the testing_set
    testing_indices = np.zeros(features_per_subject * len(test_list))
    testing_indices[0:features_per_subject] =       \
           np.arange((test_list[0]-1)* features_per_subject, 
                     test_list[0] * features_per_subject)
    if len(test_list) == 2:  
        testing_indices[features_per_subject: ] =       \
              np.arange((test_list[1]-1) * features_per_subject,   \
                        test_list[1] * features_per_subject)
    #genetate testing set
    testing_set = features[testing_indices.astype(int),:]
    testing_class = classes[testing_indices.astype(int)]
    #eliminate censored
    is_censored_testing = is_censored[testing_indices.astype(int)]
    testing_set = testing_set[~is_censored_testing]
    testing_class = testing_class[~is_censored_testing]
    
    # remainder are training set
    mask = np.isin(np.arange(0,num_feature_vectors), 
                   testing_indices.astype(int))
    #generate training set (including censored)
    training_set = features[~mask,:]
    training_class = classes[~mask]
    #eliminate censored
    is_censored_training = is_censored[~mask]
    training_set = training_set[~is_censored_training]
    training_class = training_class[~is_censored_training]
    
    
    #generate random permutations of the training and testing sets
    set_indices = np.arange(testing_set.shape[0])
    permuted_indices = np.random.permutation(set_indices)
    testing_set=testing_set[permuted_indices]
    testing_class=testing_class[permuted_indices]
    
    set_indices = np.arange(training_set.shape[0])
    permuted_indices = np.random.permutation(set_indices)
    training_set=training_set[permuted_indices]
    training_class=training_class[permuted_indices]
    
    
    return training_set, testing_set, training_class, testing_class 



def train_and_test_NN(training_data, training_class,
                            testing_data, testing_class,
                            eval_params,
                            single_freq = None,
                            plot_confusion_table=False):
    # Implements a NN with a single hidden layer using tensorflow/keras.
    # A two class model classifies a vector as being 'single_freq' 
    # versus not 'single_freq (ovr).  If single_freeq is left as None
    # a four class model is used.
    
    
    if single_freq == None:     # alias for 4-class training
        num_out_layers = 4
        test_class = testing_class
        train_class = training_class
    else:  # reassign classes as 0 or 1 for ovr
        num_out_layers = 2
        # identify class 1 (e.g. single_freq)
        training_ones = np.where(training_class == single_freq)
        testing_ones = np.where(testing_class == single_freq)
        # initiate all classes to 0 except single frequency to Class 1
        train_class = np.zeros_like(training_class)
        train_class[training_ones] = 1
        test_class = np.zeros_like(testing_class)
        test_class[testing_ones] = 1
        
    # set up NN
    model = tf.keras.models.Sequential([
         tf.keras.layers.Input((training_data.shape[1],)), 
         #hidden layer
         tf.keras.layers.Dense(100, activation='relu'),
         #tf.keras.layers.Dropout(0.2),   # randomly sets 20% of units to 0
         tf.keras.layers.Dense(num_out_layers)
         ])
    
    # set loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam', # Aplies adaptive step sizes and momentum
                  loss=loss_fn,         # to gradient descent.
                  metrics=['accuracy'])
    #train
    model.fit(training_data, train_class, epochs=10, verbose = 0)

    test_loss, test_accuracy = model.evaluate(testing_data,  
                                           test_class, verbose=0)
    #print('\nTest Accuracy:', test_accuracy)
    
    # softmax allows treatment of numeric output as probability of each
    # class
    probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
    ])
    
########################################################################    
################# How to capitalize on these probabilities for confidence
##############    in the decision???????????
    prediction_probabilities = probability_model.predict(testing_data, verbose=0)
  
    if single_freq == None:
        #print(f'average winning prob in NN 4Class = {np.mean(np.max(prediction_probabilities,  axis=-1))}')
        pred_class = np.argmax(prediction_probabilities, axis = -1)
        test_accuracy = report(testing_class, pred_class, 
                               prediction_probabilities, eval_params,
                               'NN 4Class', plot_confusion_table)
    return test_accuracy, prediction_probabilities


def train_and_test_NN_ovr(training_data, 
                                    training_class,
                                    testing_data, 
                                    testing_class, eval_params,
                                    plot_confusion_table=False):
    # ovr --> one versus the rest
    # initialize to save the results of each call to train_and_test_NN()
    probabilities = np.zeros((testing_data.shape[0],4))
    
    # trains and test each frequency as a 2 class against the other 
    # frequencies as a group  (runs a 2-class 4 times.)  Compare the 
    # probability of each class from the individual runs
    # against the others as a group and assigh to the class with the 
    # highest probabiity    
    for freq_to_test in range(4):
        accuracy, predictions = train_and_test_NN(training_data, 
                                        training_class,
                                        testing_data, 
                                        testing_class, 
                                        eval_params,
                                        single_freq=freq_to_test)
        probabilities[:,freq_to_test] = predictions[:,1]


    pred_class = np.argmax(probabilities, axis=-1)
    accuracy = np.sum(pred_class == testing_class)       \
                                /testing_class.shape[0]
                              
    report(testing_class, pred_class, probabilities, eval_params, 'NN ovr',
           plot_confusion_table)

    if False:   #for testing
        print('probs are:')
        print(probabilities[:10,:])    
        print('predicted')
        print(pred_class[:10])
        print('Truth')
        print(testing_class[:10])
   
    return accuracy



def simple_LR(training_data, training_class,
                                    testing_data, 
                                    testing_class, eval_params,
                                    plot_confusion_table=False):

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    
    # Create and train the logistic regression model
    model = LogisticRegression(max_iter=500, multi_class='ovr')
    model.fit(training_data, training_class)
    
    # Make predictions on the test set
    pred_class = model.predict(testing_data)
    
    # Calculate accuracy
    accuracy = accuracy_score(testing_class, pred_class)
    #print("Accuracy (simple LR):", accuracy)
    #print("Number of iterations:", model.n_iter_)
    
    probabilities  = 0
    report(testing_class, pred_class,probabilities, eval_params, 
           'LR ovr', plot_confusion_table)

    return accuracy




def report(testing_class, predicted_class, prediction_probabilities,
           eval_params, predictor_name, plot_confusion_table, save_to):
    '''
    This function is called at the end of training each model to calculate 
    the accuracy and electively plot a confusion table.    

    Parameters
    ----------
    testing_class : TYPE
        DESCRIPTION.
    predicted_class : TYPE
        DESCRIPTION.
    prediction_probabilities : TYPE
        DESCRIPTION.
    eval_params : TYPE
        DESCRIPTION.
    predictor_name : TYPE
        DESCRIPTION.
    plot_confusion_table : TYPE
        DESCRIPTION.
    save_to : TYPE
        DESCRIPTION.

    Returns
    -------
    accuracy : TYPE
        DESCRIPTION.

    '''

    # create confusion matrix
    c_matrix = np.zeros((4,4))
    for i in range(predicted_class.shape[0]):
        c_matrix[int(testing_class[i]), int(predicted_class[i])] += 1
    # compute accuracy  = sum of diagonals / sum of all elements  
    accuracy = np.trace(c_matrix) / np.sum(c_matrix)
    
    class_sensitivities = np.diag(c_matrix) / np.sum(c_matrix, axis=-1) 
    class_FP_rate = 1-np.diag(c_matrix) / np.sum(c_matrix, axis=0) 
    class_sensitivities = np.round(class_sensitivities,3)
    class_FP_rate = np.round(class_FP_rate,3)    
    
    
    if plot_confusion_table:
        print(f'\n\n{predictor_name}: accuracy {np.round(accuracy,3)}')
        print(f'Class sensitivities (class 0-3):{class_sensitivities}')
        print(f'      False Classification Rate (class 0-3):{class_FP_rate}')

        class_names=['8.57Hz','10Hz','12Hz','15Hz'] # name  of classes
        
        #determine figure number
        if predictor_name == 'NN 4Class':
            fignum = 1
        elif predictor_name == 'NN ovr':
            fignum = 2
        elif predictor_name == 'LR ovr':
            fignum = 3
        else:
            fignum = 4            
        fig, ax = plt.subplots(num=fignum, clear=True, figsize=(8,8))
       
        # create heatmap
        sns.heatmap(pd.DataFrame(c_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    
        # axes
        class_names=['8.57Hz','10Hz','12Hz','15Hz'] # name  of classes
        tick_marks = np.arange(len(class_names)) + 0.5
        plt.xticks(tick_marks, labels=class_names)
        plt.yticks(tick_marks, labels=class_names)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
    
        # output parameters to shorten the next text for subtitle line
        s1 = eval_params['period']
        s2 = np.round(eval_params['overlap']*100,0)
        s3 = np.round(eval_params['proportion_train']*100,0)
        if eval_params['is_random_split']:
            s4 = f'Random split: {s3}% data for training'
        else:
            s4 = f'Biased split: {s3}% data for training'
        if eval_params['saturation_criterion'] < 100:
            s5 = f"Censoring (dt = {eval_params['saturation_criterion']})"
        else:
            s5 = 'No censoring'
        #subtitle line
        plt.text(2.5,-0.05, f'{predictor_name}:   Seconds(overlap):{s1}({s2}%),  {s4},  {s5} ', ha='center')
        #title line
        plt.text(2.5, -0.15, f'Confusion Matrix: Overall accuracy {np.round(accuracy*100,1)}%', fontsize = 16, ha='center')
    
        plt.tight_layout()
        plt.show()
        
    return accuracy


def run_set(data_parameters, eval_parameters, save_to,
            periods=[2,3,4,5,6], 
            overlaps=[0.35, 0.5, 0.65, 0.8, 0.9, 0.95],
            models=['NN 4Class', 'NN ovr', 'LR'], 
            iterations = 10):
    '''
    Computes the accuracy of each model for each iteration at all the
    peroids/overlaps combinations.   It is designed to reproduce the 
    comparisons made in Acampora et. al.  The periods and overlaps are 
    those used byby them.  models is a list of our two neural network 
    models and their logistic regression model for comparison.  Using 
    the default values above this code takes an hour to run on my MacBook

    Parameters
    ----------
    data_parameters : dictionary
        DESCRIPTION.  A dictiionary describing the parameters of the 
                    data set.  See the test file (test_proj3.py) for
                    definition.
    eval_parameters : dictionary
        DESCRIPTION. A ditionary containing the elective parameters we
                use to control our models, censor epochs if we elect to,
                and otherwise evaluate the data. See the test file (test_proj3.py) for
                definition.
    save_to : String
        DESCRIPTION.  A valid path/file_name to save the results of this 
                    function.  The '.pkl' extensiion is added to save_to
    periods : list of integers
        DESCRIPTION. List of time intervals to epoch eeg_data
    overlaps : list of floats in closed interval [0.35, 0.95]
        DESCRIPTION.  List of ovelaps used in epoching successive epocchs
    models : list of strings
        DESCRIPTION.  Names of the training models 
    iterations : integer
        DESCRIPTION.  Number of iteraations through data

    Returns
    -------
    full_set : dictionary
        DESCRIPTION.  The result of are in full_accuracy array.  Other
                    are saved so the parameters of the computation are
                    known.
                full_set = {'full_accuracy_array': full_accuracy_array,
                            'eval_params': eval_parameters,
                            'periods': periods,
                            'overlaps' : overlaps,
                            'models' : models,
                            'iterations' : iterations}

    '''
    n_periods = len(periods)
    n_overlaps = len(overlaps)
    n_models = len(models)
    print(f'Running {n_periods*n_overlaps*iterations} training reps on {n_models} models')
    
    eeg_data = load_subjects(data_parameters)
    
    full_accuracy_array=np.zeros((n_periods, n_overlaps, 
                                  n_models, iterations))
    rep_counter = 0
    for period_index in range(n_periods):
        period = periods[period_index]
        eval_parameters['period'] = period
        for overlap_index in range(n_overlaps):
            overlap = overlaps[overlap_index]
            eval_parameters['overlap'] = overlap
          
            features, classes, is_censored =      \
                    get_feature_vectors(eeg_data, period, overlap, data_parameters,
                                          eval_parameters[ 'saturation_criterion'])
    
            for iter_index in range(iterations):
                rep_counter += 1
                print(f'Doing repetition number {rep_counter}')
                #randomly split training/testing each iteration
                training_data, testing_data, training_class, testing_class  =    \
                            split_feature_set(features, classes, is_censored,
                            eval_parameters['proportion_train'], data_parameters)
    
                #first model
                full_accuracy_array[period_index, overlap_index, 0, iter_index], _ =    \
                       train_and_test_NN(training_data, training_class,
                                            testing_data, 
                                            testing_class, eval_parameters)
        
                #second model
                full_accuracy_array[period_index, overlap_index, 1, iter_index] =       \
                       train_and_test_NN_ovr(training_data, 
                                                training_class,
                                                testing_data, 
                                                testing_class, eval_parameters)
        
                #third model
                full_accuracy_array[period_index, overlap_index, 2, iter_index] =        \
                       simple_LR(training_data, training_class,
                                                testing_data, 
                                                testing_class, eval_parameters)

    full_set = {'full_accuracy_array': full_accuracy_array,
                'eval_params': eval_parameters,
                'periods': periods,
                'overlaps' : overlaps,
                'models' : models,
                'iterations' : iterations}
    
    # save to disk
    with open( save_to +'.pkl', 'wb') as f:
        pickle.dump(full_set, f)

    return full_set

def load_processed_data(file_name): 
    '''
    Loads and returns an exisiting .pkl file containg a set fo processed 
    data produced by function run_set()

    Parameters
    ----------
    file_name : String
        DESCRIPTION. An exiting file produced by function run_set()

    Returns
    -------
    full_set : dictionary
        DESCRIPTION. A dictionary containg a full set as defined in 
                    run_set() above.

    '''

    with open(file_name, 'rb') as f:
        full_set = pickle.load(f)
    print(f'File {file_name} loaded ... Keys are {full_set.keys()}')
    return full_set


def plot_cross_val(full_set, model_idx,save_to=None):
    '''
    Plots and optionally saves a figure displaying accuracy with error
    bars from a series of periods and overlaps obtained with a list of 
    training models in a format that is comparable to Acampora et. al. 

    Parameters
    ----------
    full_set : dictionary
        DESCRIPTION.  Contains the specifications and testing accruacy 
                array from a list of training models. The format of 
                fullset must meet the criteria specified in the function
                run_set() (which generates a compatable dictionary.)
    method_idx : integer
        DESCRIPTION.  The index in to the models list (=fullset['methods']
                to plot in the figure.
    save_to : String, optional
        DESCRIPTION. The default is None. 'path/filename' for saved figure

    Returns
    -------
    None.

    '''
    #unpack specifications
    periods = full_set['periods']
    overlaps = full_set['overlaps']
    models = full_set['models']
    n_periods = len(periods)
    
    # unpack accuracy data and compute statistics
    # data_array is 4d   n_periods x n_overlaps x n_models x n_iterations
    data_arr = full_set['full_accuracy_array'] 
    means = np.mean(data_arr, axis = -1)
    sds = np.std(data_arr, axis=-1)
    
    # Colors for each set of periods    Note: max periods = 5
    colors = ['red', 'green', 'blue', 'black', 'orange' ]
    delta = [-0.002, -0.001,0, 0.001, 0.002]  #small offsets to avoid error
                                                #bar overlaps in zoomed figures
    fig,ax = plt.subplots(num=2, clear=True)
    
    overlap_arr = np.array(overlaps)   # x-axis in the plots below
    
    # plot full set of data
    for period_index in range(n_periods-1,0,-1): # do it backwards so that
                                            # the legend corresponds 
                                            # better with the plot lines
        dx = delta[period_index]
        ax.errorbar(overlap_arr+ dx, means[period_index, :, model_idx ], 
                     yerr=sds[period_index, :, model_idx],
                     color=colors[period_index],
                     marker= '.', label=f'{periods[period_index]} sec.',
                     linestyle='dotted')    
    plt.xticks([0.35,0.5,0.65, 0.8, 0.9, 0.95])
    plt.xlabel('Overlap Proportion')
    plt.ylabel('Proportion Correct')
    plt.ylim(0,1)
    plt.legend()

    # this is a bit clumsy.  Datum we need is a bit buried in dict 
    # structures just put censoring criterion in title of figure
    sat_crit = full_set['eval_params']['saturation_criterion'] 
    if sat_crit >= 4096:
        censored = ' (uncensored)'
    else:
        censored = f" (censored at {sat_crit})"
    plt.title(f'{models[model_idx]}: Cross Validation vs Overlap for All Periods' + censored)
    plt.grid()
    
    # replot in right lower area a zoomed image of the
    #                   high accuracy part of the above.
    # Position for the zoomed plot within the main plot
    left, bottom, width, height = 0.5, 0.15, 0.4, 0.35
    ax_zoom = fig.add_axes([left, bottom, width, height])
    
    for period_index in range(n_periods-1,0,-1): 
        dx = delta[period_index]
        ax_zoom.errorbar(overlap_arr+ dx, means[period_index, :, model_idx ], 
                     yerr=sds[period_index, :, model_idx],
                     color=colors[period_index],
                     marker= '.', label=f'{periods[period_index]} Sec.',
                     linestyle='dotted')    
    
    # want grid lines, but not x-labels    
    plt.xticks(ticks = [ 0.9, 0.95], labels = [' ', ' '])
    # region to plot
    plt.xlim(0.89, 0.96)  
    plt.ylim(0.8,0.975)
    # label grid lines inside zoomed plot
    plt.text(0.9,0.81,'0.9', ha = 'center')
    plt.text(0.95,0.81,'0.95', ha = 'center')
    plt.grid()

    plt.show()

    if not (save_to == None):
        plt.savefig(save_to)
    
    return
