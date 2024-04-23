# BCIs_Project3
Written 4/25/24 by Ronald Bryant, Sadegh Khodabandeloo, and Marit Scott 

## Dataset
The dataset is described by Acampora et al. 2021. https://pubmed.ncbi.nlm.nih.gov/33659590/. 
The data can be downloaded from: https://data.mendeley.com/datasets/px9dpkssy8/draft?a=7140665d-a0f0–40b2-a9fd-a731d21b6222   (Requires an account to download.)

This dataset contains raw eeg data related to SSVEP signals acquired from eleven volunteers by using acquisition equipment based on a single-channel dry-sensor recording device. The recorded EEG data from a single volunteer contains the response to an intermittent source of light, which is sequentially emitted at four different frequencies, namely 8.57 Hz (F1), 10 Hz (F2), 12 Hz (F3), or 15 Hz (F4). Each recording has a duration of 16s and is sampled at 256 Hz. There are 11 csv files in this dataset from 11 subjects. Each csv files contains 4096 rows (sequential time points) and a column that includes four voltage readings separated by semicolons (';'), one for each of the four stimulus freqeuncies at a time point (row).  Voltage readings ranging from 0 to 1023 represent a dynamic range of +/-0.39 mV.  The  ADC that digitized the signal has an unspecified internal gain.

## Loading the Data
The data files are labeled '.csv' but are not truely comma separated.  They are text files with semicolon separated values.  We use the python library pandas for importing the files into a pandas dataframe. Then each row of the data frame is converted to numeric values that are ultimately stored in a numpy array of shape (4096,4).

The following lines of python code create the numpy array for the subject number 11.
#######################################################################
import pandas as pd
import numpy as np

#load data file
df = pd.read_csv('subject11.csv') #load data to the dataframe df

#initialize numpy array to hold eeg data for each frequency. 
len_eeg = df.shape[0]
num_freqs = 4       # based on descripition of the data files
eeg = np.zeros((len_eeg, num_freqs)) # initialize array 

#loop through each row of dataframe, convert to number and populate array
for time_index in range(len_eeg): 
    line_str = df.iloc[i].values  #returns string containing data
    str_list = line_str[0].split(';')  #split string at semicolons
    eeg[i,:] = [float(x) for x in str_list] #convert strings to numbers
                                            #and populates the eeg data array 
#########################################################################
