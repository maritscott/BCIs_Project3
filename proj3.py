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
        DESCRIPTION. The default is 'All'.  If an integer from the closed
                        set [0,11] is entered, only that individuals data
                        is loaded and the EEG data is plotted.
    zoom : boolean    default False
        DESCRIPTION.  If set to True and a single subject is selected, the
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
    
    def load_one(subject_number):
        '''
        The file for the subject is read and the eeg at each stimulus
        frequency is returned.

        Parameters
        ----------
        subject_number : integer from the closed set [0, 11]
            DESCRIPTION.  Data for this subject is loaded

        Returns
        -------
        eeg : 2d array of float    num_freqs x num_times
            DESCRIPTION. A subject's EEG from num_freqs stimulation 
                        frequencies. The EEG values are integers in the 
                        closed interval [0, 1023].  The sampling is 
                        4096 time points at 256 Hz = 16 seconds. 
        '''
        
        # Load the data using a semicolon as the separator
        df = pd.read_csv(f'subject{subject_number}.csv', sep=';') 

        # Transpose to a numpy array of integers in the shape (num_freq x num_times)
        eeg = df.values.astype(int).T

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
                            load_one(sub_num+1)
                            
                            
    else:  # load indicated subject and plot EEG for each stimulus
        eeg_array = load_one(subject)
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
    it's ADC for more than  `criterion`  consequtive time points

    Parameters
    ----------
    epoch : TYPE     1d array of integers 
        DESCRIPTION.   EEG values in close interval [0, 1023] from the ADC
    criterion : TYPE  Integer
        DESCRIPTION.   If  `criterion`  0's or 1023's appear sequentially in
                the epoch the function returns True  (else False)

    Returns
    -------
    bool
        DESCRIPTION.   True if  `criterion`  or more 0's or 1023's appear sequentially
                in an epoch. (else False)

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
    Helper function for get_feature_vectors()
    Returns epochs of eeg data, each with a length of `time_period`,
    starting from the beginning of `eeg_data` with overlapping of 
    time period of proportion `overlap`.

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
        DESCRIPTION.  percentage that intervals may overlap in decimal form.
                    With a large overlap many epochs can be obtained from the 
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
        DESCRIPTION.  for each subject an EEG from num_freqs stimulation 
                    frequencies The EEG values are in arbitrary units since
                    an unknown amplification occurs during acquisition.
                    The sampling is 4096 point at 256 Hz = 16 seconds.
    period : float from the open interval (0, 16)
        DESCRIPTION.  Duration of each epoch
    overlap : float    from open interval (0, 1)
        DESCRIPTION.  amount intervals may overlap.  With a large 
                    overlap many epochs can be obtained from the 
                    16 second EEG even with a relatively large time_period
    params : Dictionary    
        DESCRIPTION.  Contains basic parameters describing the data files.
                    See the test file for specifics for this data set. 
                    Includes fs (sampling freqeuncy)
    saturation_criterion : Integer (optional with default of 4)
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


def report(testing_class, predicted_class, prediction_probabilities,
           eval_params, predictor_name, plot_confusion_table):

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