# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:59:45 2024

@author: MA11201
"""
import os
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 170
import seaborn as sns


columns =['Ipv', 'Vpv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Iabc', 'If', 'Vabc', 'Vf']
def data_read(dir_path, data_type, start, end):
    """
    Parameters
    ----------
    dir_path : data files path
        indicates the path of the folder containing the csv files.
    data_type : string
        indicates the type of data one needs to create (train, validation, test, fault).
    start : int
        indicates the start of a data file.
    end : int
        indicates the end of a data file.

    Returns
    -------
    pandas DataFrame for train, validation, test
    tensor for fault data.

    """
    data = []
    
    for file in sorted(os.listdir(dir_path)):
        data_file = pd.read_csv(os.path.join(dir_path, file))
        data_file = data_file.drop(['Unnamed: 0'], axis=1)
        data.append(data_file)
        
    scaler = StandardScaler()
    scaler.fit(data[0])
    scaled_data = scaler.transform(data[0])
    scaled_data = pd.DataFrame(scaled_data, columns = data[0].columns)
    
    if data_type == 'train':     
        return scaled_data.iloc[start:end, :]        
    elif data_type == 'val':
        return scaled_data.iloc[start:end, :]  
    elif data_type == 'test':
        return scaled_data.iloc[start:end, :]    
    elif data_type == 'fault':
        fault_data = []
        for i in range(1, 8):
            fdata_scaled = scaler.transform(data[i])
            fdata_split = fdata_scaled[start:end, :]
            fdata_tensor = torch.tensor(fdata_split, dtype=torch.float)
            fault_data.append(fdata_tensor)
            
        return fault_data
    else:
        return print('data type is invalid')

#edge indices and weights generatio for the dataset

def graph_attr(data, sim_met):
    """
    Parameters
    ----------
    data : pandas data frame
        the data file for generating edge indices and edge weights.
    sim_met : string
        indicates the similarity measurement technique among the features.

    Returns
    -------
    edge_index : tensor
        edge indices for the graph structured data.
    edge_weight : tensor
        edge weights for the graph structured data.

    """
           
    dim = data.shape[1]
    edge = [[], []]
    for j in range(dim):
        for k in range(dim):
            if j != k:
                edge[0].append(j)
                edge[1].append(k)
                
    edge_index = torch.tensor(edge, dtype=torch.long)
    
    sim_val = []
    cols = data.columns
    for c1 in range(len(cols)):
        vec_a = np.array(data[cols[c1]])
        for c2 in range(len(cols)):
            if c2!=c1:
                vec_b = np.array(data[cols[c2]])
                if sim_met == 'Cosine':    
                    cos_sim = np.abs(dot(vec_a, vec_b)/(norm(vec_a)*norm(vec_b)))
                    sim_val.append(cos_sim)
                else:
                    euclidean = np.linalg.norm(vec_a-vec_b)
                    sim_val.append(euclidean)
                    
    edge_weight = torch.tensor(sim_val, dtype=torch.float)
    
    return edge_index, edge_weight
                    
                
def recon_perf(true_data, pred_data):
    result = {}
    mse = mean_squared_error(true_data, pred_data)
    mae = mean_absolute_error(true_data, pred_data)
    mape = mean_absolute_percentage_error(true_data, pred_data)
             
    result['mse'] = mse
    result['mae'] = mae
    result['mape'] = mape
    
    return result

def ConfidenceInterval(data, conf_value):
    mean, std = data.mean(), data.std(ddof=1)
    lower, higher = stats.norm.interval(conf_value, loc=mean, scale=std)
    return lower, higher


def SPEControlLimit(tr_data, pred_data, conf_val):
    hist=[]
    for i in range(tr_data.shape[0]):
        SPE = np.matmul((tr_data-pred_data)[i, :].T, (tr_data-pred_data)[i, :])
        hist.append(SPE)
    hist1 = np.array(hist)
    low_lim, high_lim = ConfidenceInterval(hist1, conf_val)
    return hist1, low_lim, high_lim

def latent_vars(model, data, edge_idx, edge_wt):
    pred = model(data, edge_idx, edge_wt)
    hv = model.hidden_vars
    hv = hv.detach().numpy()
    return hv


def eval_normal(model, data, conf_val, eval_type):
    
    pred = model(data)
    
    pred1 = pred.detach().numpy()
    data1 = data.detach().numpy()
    recon_result = recon_perf(data1, pred1)
    error, low, high = SPEControlLimit(data1, pred1, conf_val)
    if eval_type == 'train':
        return error, recon_result, pred1, data1
    elif eval_type == 'val':
        return error, recon_result, high, pred1, data1
    elif eval_type == 'test':
        return error, recon_result, pred1, data1
    else:
        return print('invalid evaluation type')
    
def eval_fault(model, data, conf_val, eval_type):
    if eval_type == 'fault':
        fault_pred = []
        fault_data = []
        recon_result = []
        fault_error = []
        for i in range(len(data)):
            pred = model(data[i])
            pred1 = pred.detach().numpy()
            data1 = data[i].detach().numpy()
            fault_pred.append(pred1)
            fault_data.append(data1)
            error,_,_ = SPEControlLimit(data1, pred1, conf_val)
            fault_error.append(error)
            recon_result.append(recon_perf(data1, pred1))
        return fault_error, recon_result, fault_pred, fault_data
            
    else:
        return print('invalid evaluation type')

def SPE(y, y_pred):
   error = (y-y_pred)**2
   error1 = pd.DataFrame(error, columns = columns)
   return error, error1   

def error_cont(pred_data, true_data):
    feature_mse = {}
    for i in range(13):        
        mse = mean_squared_error(true_data[:, i], pred_data[:, i])     
        feature_mse[columns[i]] = mse
    return feature_mse

def error_plot(pred_data, true_data, fault_type):
    data = error_cont(pred_data, true_data)
    x = data.keys()
    y = data.values()
    fig, ax = plt.subplots(figsize = (10, 5))
    # creating the bar plot
    ax.bar(x, y, color ='blue', width = 0.4)
     
    ax.xlabel("Features")
    plt.ylabel("MSE")
    plt.title("{}".format(fault_type))
    plt.show()

    
                
def plot_recon(data, threshold, title, fig_save=False):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(data, color = 'steelblue', label=title)
    ax.axhline(threshold, color = 'mediumblue', ls = '-.', label = 'Threshold')
    ax.set_xlabel('data points')
    ax.set_ylabel('SPE')
    ax.grid(visible=True, which='major', axis='both')
    ax.legend()
    if fig_save ==True:
        plt.savefig(r"C:\a_PhD Research\FDD\Codes\GCN_VAE1\save_dir\Figures\ReconFig_{}.pdf".format(title), bbox_inches='tight')
    plt.tight_layout()
    plt.show()


    


def plot_result2(test_df, val_df, limits):

  fig, axs= plt.subplots(nrows=4, ncols=2, figsize=(10,6)) #(width, height)
  plt.subplots_adjust(hspace=0.3) #Vertical
  plt.subplots_adjust(wspace =0.3) #Horizontal

  for c, ax in zip(columns, axs.ravel()):
    ax.plot(test_df[c], color='red')
    ax.plot(val_df[c], color='green')
    ax.axhline(limits[c], color='black', ls='--')
    ax.set_xlabel(c)
    ax.set_ylabel('re_error')
    ax.set_yscale('log')
  plt.show()
  return None              



def perf(error, threshold):
    error = error.tolist()
    count = 0
    for value in range(len(error)):
        if error[value]>threshold:
            count = count +1
    far  = count/(len(error))
    return far    
            
            
            
            
            


