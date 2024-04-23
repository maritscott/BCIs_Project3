#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:08:45 2024

@author: marit scott, sadegh khodabandeloo, ron bryant
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
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

    Returns
    -------
    eeg_array : 2d array of float    num_eegs x num_times
        DESCRIPTION.  EEG from the selected subject (or all subjects)
                    at 4 frequencies are returned. Each subjects EEGS 
                    are in adjacent rows. 
                    The EEG values are in arbitray units since an
                    unknown amplification occurs during acquisition.
                    The sampling is 4096 point at 256 Hz = 16 seconds.
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
            DESCRIPTION.  an EEG from num_freqs stimulation frequencies 
                        The EEG values are in arbitray units since an
                        unknown amplification occurs during acquisition.
                        The sampling is 4096 point at 256 Hz = 16 seconds. 
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
            eeg[:,df_index] = [float(x) for x in str_list]
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


   

def epoch_data(eeg_data, time_period, overlap, parameters):
    '''
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
 
    return  epochs, epoch_classes


def get_feature_vectors(eeg_data, period, overlap, params):
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
                      n_epochs_per_eeg in each class (all adjacent rows).   '''
    
    epochs, vector_classes = epoch_data(eeg_data, period, overlap, params)

    # fft of epochs
    ffts = abs(np.fft.rfft(epochs,axis=-1))
    
    # select range of frequencies for feature vector
    df = 1/period  # frequency resolution
    freq_lower = 4
    freq_higher = 32 
    index_lower = int(round(freq_lower/df))
    index_higher = int(round(freq_higher/df))
    
    features  = ffts[:,index_lower:index_higher]
    #noramlize
    feature_vectors = (features - features.mean(axis=-1, keepdims=True) )   \
                    / features.std(axis=-1,keepdims=True)
    
    return feature_vectors, vector_classes  

def split_feature_set(features, classes, proportion_train, parameters):
    '''
    Randomly splits the feature vectors into training sets and testing sets

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
    
    
    #generate a random permutation of the feature vectors
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

def biased_split_feature_set(features, classes, parameters, test_list):
    '''
    Returns a biased testing set of the all epochs from the 1 or 2 
    subjects in the test_list

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
 
    #This few lines take advantage of the ordering of the feature vectors
    #in the array to select the subjects in test_list to be the testing_set
    testing_indices = np.zeros(features_per_subject * len(test_list))
    testing_indices[0:features_per_subject] =       \
           np.arange((test_list[0]-1)* features_per_subject, 
                     test_list[0] * features_per_subject)
    if len(test_list) == 2:
        testing_indices[features_per_subject: ] =       \
              np.arange((test_list[1]-1) * features_per_subject,   \
                        test_list[1] * features_per_subject)
    
    testing_set = features[testing_indices.astype(int),:]
    testing_class = classes[testing_indices.astype(int)]
    
    mask = np.isin(np.arange(0,num_feature_vectors), 
                   testing_indices.astype(int))
    training_set = features[~mask,:]
    training_class = classes[~mask]
    
    
    #generate permutations of the training and testing sets
    set_indices = np.arange(testing_set.shape[0])
    permuted_indices = np.random.permutation(set_indices)
    testing_set=testing_set[permuted_indices]
    testing_class=testing_class[permuted_indices]
    
    set_indices = np.arange(training_set.shape[0])
    permuted_indices = np.random.permutation(set_indices)
    training_set=training_set[permuted_indices]
    training_class=training_class[permuted_indices]
    
    
    return training_set, testing_set, training_class, testing_class 



def train_and_test_LR(training_data, training_class,
                            testing_data, testing_class, single_freq = None):
    # this implements a logistic regression two class model using 
    # keras/tensor flow that classifies a vector as being 'single_freq' 
    # versus not 'single_freq.  If single_freeq is left as None
    # a four class model is used.
    
    if single_freq == None:
        num_out_layers = 4
        test_class = testing_class
        train_class = training_class
    else:  # reassign classes as 0 or 1
        num_out_layers = 2
        training_ones = np.where(training_class == single_freq)
        testing_ones = np.where(testing_class == single_freq)
        train_class = np.zeros_like(training_class)
        train_class[training_ones] = 1
        test_class = np.zeros_like(testing_class)
        test_class[testing_ones] = 1
        
    
    model = tf.keras.models.Sequential([
         tf.keras.layers.Input((training_data.shape[1],)), # instead of below
         #tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(100, activation='relu'),
         #tf.keras.layers.Dropout(0.2),   # randomly sets 20% of units to 0
         tf.keras.layers.Dense(num_out_layers)
         ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(training_data, train_class, epochs=10)

    test_loss, test_accuracy = model.evaluate(testing_data,  
                                           test_class, verbose=2)
    print('\nTest Accuracy:', test_accuracy)

    probability_model = tf.keras.Sequential([
      model,
      tf.keras.layers.Softmax()
    ])
    prediction_probabilities = probability_model.predict(testing_data)
 
    return test_accuracy, prediction_probabilities


def train_and_test_LR_ovr(training_data, 
                                    training_class,
                                    testing_data, 
                                    testing_class):
    # ovr --> one versus the rest
    # initialize
    probabilities = np.zeros((testing_data.shape[0],4))
    
    # traina and test each frequency as a 2 class against the other 
    # frequencies as a group  (runs a 2-class 4 times.)  Compare the 
    # probability of each class from the individual runs
    # against the others as a group and assigh to the class with the 
    # highest probabiity    
    for freq_to_test in range(4):
        accuracy, predictions = train_and_test_LR(training_data, 
                                        training_class,
                                        testing_data, 
                                        testing_class, 
                                        single_freq=freq_to_test)
        probabilities[:,freq_to_test] = predictions[:,1]

    pred_class = np.argmax(probabilities, axis=-1)
    accuracy = np.sum(pred_class == testing_class)       \
                                /testing_class.shape[0]
                              
    report(testing_class, pred_class)

    if False:   #for testing
        print('probs are:')
        print(probabilities[:10,:])    
        print('predicted')
        print(pred_class[:10])
        print('Truth')
        print(testing_class[:10])
   
    return accuracy


def report(testing_class, predicted_class):
    import seaborn as sns

    c_matrix = np.zeros((4,4))
    for i in range(predicted_class.shape[0]):
        c_matrix[int(testing_class[i]), int(predicted_class[i])] += 1

    
    class_names=[0,1,2,3] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(c_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.Text(0.5,257.44,'Predicted label');


    
