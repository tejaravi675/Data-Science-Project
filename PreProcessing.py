import numpy as np
from numpy import savetxt
import pandas as pd
import os


def create_features(sample, label):
    # Clean data
    if np.max(sample) >= 1000:
        label = -1
    if len(sample) < 20:
        label = -1
    if np.min(sample) < 0.5*np.median(sample) or np.max(sample) > 1.5*np.median(sample):
        label = -1

    # Calculate features
    results = np.array([label, len(sample), np.min(sample),np.max(sample),np.median(sample), np.mean(sample), np.std(sample)])
    
    return results

   
def combine_data(no_file):
    # Locate specific file
    no_file = 2  # Number of data file from database
    
    local_dir = os.path.dirname(__file__)  # Set directory same as python file
    path_data = (local_dir +'\Data\ECG_data\Data'+ str(no_file) +'.txt')
    path_label = (local_dir +'\Data\Class\Control'+ str(no_file) +'.txt')
    
    # Load data
    data = pd.read_csv(path_data, header = None, usecols = [0,1], sep = ' ')
    label = pd.read_csv(path_label, header = None, delim_whitespace=True)
    
    # Add extra empty row to data (looking forward in loop)
    label = label.append(pd.DataFrame(['23:59:59',-1]).T) 
    data = data.append(pd.DataFrame(['00:00:00',0]).T)
     
    # Combine data and classes
    j = 0
    sample = np.empty(0)
    features = np.zeros((label.shape[0]-1,7))
    
    for i in range(0, data.shape[0]-1): 
        # Make exception when time has transition from PM to AM
        if label.iloc[j+1,0][:-4] < '00:00:30' and data.iloc[i,0] > '23:59:30':
            transition_time = True
        else:
            transition_time = False
        
        # Combine all values of one 30 sec period 
        sample = np.append(sample,data.iloc[i,1]) 
        
        # Calculate features when sample is complete
        if data.iloc[i+1,0] >= label.iloc[j+1,0][:-4] and transition_time == False:
            properties = create_features(sample, label.iloc[j,1]).T  # If row with values is filled, calculate the properties of this sample
            features[j,:] = properties  # Add features properties according to 30 sec label period
            sample = []  # Reset current list
            j = j+1  # Go to next 30 sec sample 
    
    # Remove unusefull data
    features = features[np.logical_not(features[:,0] == -1)]  # Remove all labels with -1 (unvalid data)
    features = features[:-1,:]  # Remove empty row at end
    
    return features  

   

# Collect features from all files    
total_files = 804

df = np.empty((0,7), int)
for no in range(1,total_files):
    features = combine_data(no)
    df = np.append(df, features, axis=0)
    print(round(no/total_files*100,1), '% complete')  # Update progress

# save features as csv file
savetxt('features training.csv', df, delimiter=';')







