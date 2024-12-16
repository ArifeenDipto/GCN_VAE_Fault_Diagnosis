# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:12:30 2024

@author: MA11201
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.preprocessing import MinMaxScaler
folder_path = r"C:\a_PhD Research\FDD\Dataset\IPPT"

#reading data files
data_list = []
for file in sorted(os.listdir(folder_path)):
    df = pd.read_csv(os.path.join(folder_path, file))
    df = df.drop(['Time'], axis=1)
    #df = df.iloc[:20000, :]
    data_list.append(df)
    
#Apply convolutional filtering to reduce noise
def apply_convolution(sig, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered

#Apply boxplot to remove Outliers 
def box_outlier(data):
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1    #IQR is interquartile range.
  lower_bound =Q1 - 1.5 * IQR
  upper_bound =Q3 + 1.5 *IQR
  indices_lower = np.where(data <= lower_bound)[0] 
  indices_upper = np.where(data >= upper_bound)[0]
  data.drop(index = indices_lower, inplace=True)
  data.drop(index = indices_upper, inplace=True)
  return data    


#Noise free data
conv_data = []
for i in range(len(data_list)):
    noise_free = data_list[i].apply(lambda srs: apply_convolution(srs, 35))
    conv_data.append(noise_free)

fig, ax = plt.subplots(figsize=(5, 3))
plt.plot(conv_data[0].iloc[:100, 1])



for i in range(len(conv_data)):
    for c in conv_data[0].columns:
        new_data = box_outlier(conv_data[i][c])
        conv_data[i][c] = new_data
        
for i in range(len(conv_data)):
    conv_data[i].dropna(inplace=True)
#conv_data[1].isna().values.sum()







for i in range(len(conv_data)):
    print(f'data_{i}\t shape={conv_data[i].shape}')


#Visualization
for i in range(1, len(conv_data)):
    fig, axs= plt.subplots(nrows=5, ncols=3, figsize=(10,11)) #(width, height)
    plt.subplots_adjust(hspace=0.5) #Vertical
    plt.subplots_adjust(wspace =0.5) #Horizontal
    features = conv_data[0].columns
    for c, ax in zip(range(13), axs.ravel()):
        ax.plot(conv_data[0][features[c]], color='green')
        ax.plot(conv_data[i][features[c]], color ='orange')
        #ax.set_title(features[c])
        ax.set_xlabel('time: {}'.format(i))
        ax.set_ylabel(features[c])  
plt.show()


#Preprocessed data storing
save_folder = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_35"
for i in range(len(conv_data)):
    file_name = 'data_{}.csv'.format(i)
    conv_data[i].to_csv(os.path.join(save_folder, file_name))






data_folder = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_5"
files = []
for file in sorted(os.listdir(data_folder)):
    data_csv =  pd.read_csv(os.path.join(data_folder, file))
    files.append(data_csv)


files1 = []
for i in range(len(files)):
    data1 = MinMaxScaler().fit_transform(files[i])
    data2 = pd.DataFrame(data1, columns = files[0].columns)
    files1.append(data2)


fig, ax =  plt.subplots(figsize=(5, 3))
sns.kdeplot(files1[0].iloc[:, 3], label ='N')
sns.kdeplot(files1[5].iloc[:, 3], label ='F5')
# =============================================================================
# sns.kdeplot(files1[2].iloc[:, 1], label ='F2')
# sns.kdeplot(files1[3].iloc[:, 1], label ='F3')
# sns.kdeplot(files1[4].iloc[:, 1], label ='F4')
# sns.kdeplot(files1[5].iloc[:, 1], label ='F5')
# sns.kdeplot(files1[6].iloc[:, 1], label ='F6')
# sns.kdeplot(files1[7].iloc[:, 1], label ='F7')
# =============================================================================
ax.legend()
plt.show()



















