# BCIs_Project3
Written 4/25/24 by Ronald Bryant, Sadegh Khodabandeloo, and Marit Scott 

## Dataset

Dataset described by Acampora et al. 2021. Paper downloaded from https://pubmed.ncbi.nlm.nih.gov/33659590/.
This dataset contains raw eeg data related to SSVEP signals acquired from eleven volunteers by using an acquisition equipment based on a single-channel dry-sensor recording device. The recorded EEG data from a single volunteer contains the response to an intermittent source of light, which is emitted at four different frequencies, namely 8.57 Hz (F1), 10 Hz (F2), 12 Hz (F3) and 15 Hz. Each freq stimulus has duration of 16s and the sampling freq is 256. There are 11 csv files in this dataset from 11 subjects. Each csv files contains 4096 rows (number of samples) and 1 column including 4 numbers (4 stimuli frequencies).

## Loading the Data
Since the data is in csv format, we use one of the python's library dedicated to reading csv files called panda. Thus, importing panda to load the dataset is required.
At first the fillowing function should be used for separating 4 columns. 
import pandas as pd

#get data
data = pd.read_csv('subject11.csv') #extracting the data from the csv file for subject 11 for example
len_eeg = data.shape[0]
data_new = np.zeros((len_eeg,4)) #creating a matrix of zeros
for i in range(len_eeg): # looping through each sample to separate 4 numbers to 4 different column
    line = data.iloc[i].values
    lineslist = line[0].split(';')
    data_new[i,:] = [float(x) for x in lineslist]
#And at the end the raw data should be divided by 500 for scaling purposes
data_final = data_new/500   #arbitrary scalling



## Notes
(additional information about how the dataset is organized, what the rows and columns represent, etc.)
