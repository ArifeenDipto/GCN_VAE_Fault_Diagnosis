# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:42:45 2024

@author: MA11201
"""


import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
folder_path = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_35"

data = []
for files in sorted(os.listdir(folder_path)):
    csv_files = pd.read_csv(os.path.join(folder_path, files))
    csv_files1 = csv_files.drop(['Unnamed: 0'], axis=1)
    data.append(csv_files1)
    
#correlation plot
corr_mat = data[0].corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(abs(corr_mat), annot=False, cmap= 'crest', ax = ax)
print(abs(corr_mat))
    
#cosine plot
sim_val = np.empty(shape=(13, 13))
cols = data[0].columns
for c1 in range(len(cols)):
    vec_a = np.array(data[0][cols[c1]])
    for c2 in range(len(cols)):
        #if c2!=c1:
        vec_b = np.array(data[0][cols[c2]])                
        cos_sim = np.abs(dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b)))
        sim_val[c1][c2] = cos_sim
            
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(abs(sim_val), annot=False, cmap= 'crest', ax = ax)


#cosine plot with normalization
sim_val = np.empty(shape=(13, 13))
cols = data[0].columns
for c1 in range(len(cols)):
    vec_a = np.array(data[0][cols[c1]])
    di = []
    for c2 in range(len(cols)):
        #if c2!=c1:
        vec_b = np.array(data[0][cols[c2]])                
        cos_sim = np.abs(dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b)))
        di.append(cos_sim)
    beta = np.mean(di)
    di = [np.exp(-x/beta) for x in di]
    di = [x/np.sum(di) for x in di]
    sim_val[c1] = di            
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(abs(sim_val), annot=False, cmap= 'crest', ax = ax)


#Euclidean plot
euc_val = np.empty(shape=(13, 13))
cols = data[0].columns
for c1 in range(len(cols)):
    vec_a = np.array(data[0][cols[c1]])
    for c2 in range(len(cols)):
        #if c2!=c1:
        vec_b = np.array(data[0][cols[c2]])                
        euclidean = np.linalg.norm(vec_a - vec_b)
        euc_val[c1][c2] = euclidean
            
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(abs(euc_val), annot=False, cmap= 'crest', ax = ax)

#Euclidean plot with normalization
euc_val = np.empty(shape=(13, 13))
cols = data[0].columns
for c1 in range(len(cols)):
    vec_a = np.array(data[0][cols[c1]])
    di = []
    for c2 in range(len(cols)):
        #if c2!=c1:
        vec_b = np.array(data[0][cols[c2]])                
        euclidean = np.linalg.norm(vec_a - vec_b)
        di.append(euclidean)
    beta = np.mean(di)
    di = [np.exp(-x/beta) for x in di]
    di = [x/np.sum(di) for x in di]
    euc_val[c1] = di
            
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(abs(euc_val), annot=False, cmap= 'crest', ax = ax)





# =============================================================================
#             else:
#                 euclidean = np.linalg.norm(vec_a-vec_b)
#                 sim_val.append(euclidean)
# =============================================================================
    
# =============================================================================
# cols = data[0].columns
# for c in cols:
#     a = np.array(data[0][c])
#     b = np.array(data[2][c])
#     ab = [a, b]
#     fig, ax = plt.subplots(figsize=(5, 3))
#     sns.boxplot(ab, ax=ax)
#     ax.set_title(c)
#     #plt.savefig(r"C:\a_PhD Research\FDD\IPPT Exp\Figures\Fault7\Feat_{}.svg".format(c), bbox_inches='tight')
#     #plt.tight_layout()
#     plt.show()
# =============================================================================



def cdf(data):
  x=np.array(data)
  count, bins_count = np.histogram(x, bins=10)
  pdf = count/ sum(count)
  cdf = np.cumsum(pdf)
  return bins_count, cdf, pdf


cols = data[0].columns
for c in cols:
    a=data[0][c]
    b=data[7][c]
    ks2test = stats.ks_2samp(a, b)
    print(f'{c}\t{ks2test}')
    a_res = cdf(a)
    b_res = cdf(b)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(a_res[0][1:], a_res[1], color="black", label="NF")   
    ax.plot(b_res[0][1:], b_res[1], color="blue", label="F1")
    ax.set_title(c)
    #plt.savefig(r"C:\a_PhD Research\FDD\IPPT Exp\Figures\CDF\Fault7\{}.svg".format(c), bbox_inches='tight')
    #plt.tight_layout()
    plt.show()
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


