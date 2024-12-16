# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:20:10 2024

@author: MA11201
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

folder_path = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_35"


data = []
for files in sorted(os.listdir(folder_path)):
    file = pd.read_csv(os.path.join(folder_path, files))
    file1 = file.drop(['Unnamed: 0'], axis=1)
    data.append(file1)

cols = data[0].columns

data1 = []
for i in range(len(data)):
    dt = MinMaxScaler().fit_transform(data[i])
    dt1 = pd.DataFrame(dt, columns=cols)
    data1.append(dt1)
    
# =============================================================================
# pca = PCA(n_components=2)
# 
# dt0_pca = pca.fit_transform(data1[0].iloc[:7000, :])
# dt0_pca.shape
# 
# dt1_pca = pca.fit_transform(data1[1].iloc[:1000, :])
# dt1_pca.shape
# 
# dt2_pca = pca.fit_transform(data1[2].iloc[:1000, :])
# dt2_pca.shape
# 
# dt3_pca = pca.fit_transform(data1[3].iloc[:1000, :])
# dt3_pca.shape
# 
# dt4_pca = pca.fit_transform(data1[4].iloc[:1000, :])
# dt4_pca.shape
# 
# dt5_pca = pca.fit_transform(data1[5].iloc[:1000, :])
# dt5_pca.shape
# 
# dt6_pca = pca.fit_transform(data1[6].iloc[:1000, :])
# dt6_pca.shape
# 
# dt7_pca = pca.fit_transform(data1[7].iloc[:1000, :])
# dt7_pca.shape
# 
# 
# 
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.scatter(dt0_pca[:, 0], dt0_pca[:, 1])
# ax.scatter(dt1_pca[:, 0], dt1_pca[:, 1])
# ax.scatter(dt2_pca[:, 0], dt2_pca[:, 1])
# ax.scatter(dt3_pca[:, 0], dt3_pca[:, 1])
# ax.scatter(dt4_pca[:, 0], dt4_pca[:, 1])
# ax.scatter(dt5_pca[:, 0], dt5_pca[:, 1])
# ax.scatter(dt6_pca[:, 0], dt6_pca[:, 1])
# ax.scatter(dt7_pca[:, 0], dt7_pca[:, 1])
# plt.show()
# 
# =============================================================================

df_new = []
for i in range(len(cols)):
    df_cols = pd.DataFrame()
    for j in range(len(data1)):
        dff = data1[j][cols[i]]
        df_cols['{m}_{n}'.format(m = cols[i], n = j)] = dff        
    df_new.append(df_cols)


for i in range(len(df_new)):
    fig, ax = plt.subplots(figsize=(5, 3))
    #sns.boxplot(df_new[i], medianprops=dict(color="yellow", alpha=0.7), ax=ax)
    sns.kdeplot(df_new[i], ax=ax)











