# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:28:40 2024

@author: MA11201
"""

import sys
sys.path.append(r'C:\a_PhD Research\FDD\IPPT Exp\AE')

from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from AE_Model import AE
from AE_utils import *
import warnings
warnings.filterwarnings("ignore")

data_folder = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_5" #folder path of data
model_path = r"C:\a_PhD Research\FDD\IPPT Exp\AE\save_dir\AE_wind_5.pkl" # pretrained model path

#loading the pretrained model
model = AE(13)
model.load_state_dict(torch.load(model_path))

#creating normal datasets
X_tr = data_read(data_folder, 'train', 0, 7000)
X_val = data_read(data_folder, 'val', 7000, 9000)
X_test = data_read(data_folder, 'test', 9000, 10000)

#X_te2 = data_read(data_folder, 'test', 15000, 16000)
#X_te3 = data_read(data_folder, 'test', 16000, 16000)

X_tr.shape, X_val.shape, X_test.shape

# =============================================================================
# fig, axs= plt.subplots(nrows=4, ncols=2, figsize=(11,10)) #(width, height)
# plt.subplots_adjust(hspace=0.3)
# plt.subplots_adjust(wspace =0.3)
# 
# features = X_tr.columns
# 
# 
# for c, ax in zip(features, axs.ravel()):
#   ax.plot(X_tr[c], label=c)
#   ax.legend()
# plt.show()
# =============================================================================


#creating fault datasets
X_fault = data_read(data_folder, 'fault', 0, 1000)

#creating graph attributes

X_tr = torch.tensor(np.array(X_tr), dtype=torch.float)
X_val = torch.tensor(np.array(X_val), dtype=torch.float)
X_test = torch.tensor(np.array(X_test), dtype=torch.float)



    





#reconstruction plot
# =============================================================================
# xtr = model(X_tr, edge_indices, edge_weights)
# xtrp = xtr[0].detach().numpy()
# xtra = X_tr.detach().numpy()
# fig, ax = plt.subplots(figsize=(5,3))
# xtrp.shape
# =============================================================================


#evaluation
model = model.eval()
conf_value = 0.99
#validation
val_re = eval_normal(model, X_val, conf_value, 'val')
plot_recon(val_re[0], val_re[2], 'Validation', False)
val_re[1]
#Normal test
te_re1 = eval_normal(model, X_test, conf_value, 'test')
plot_recon(te_re1[0], val_re[2], 'Normal_test', False)
te_re1[1]




#False alarm rate
res_val = perf(val_re[0], val_re[2])
res_val*100
res_test = perf(te_re1[0], val_re[2])
res_test*100




#Fault data
fault_re = eval_fault(model, X_fault, conf_value, 'fault')


plot_recon(fault_re[0][0], val_re[2], 'Inverter', False)
res_fault = perf(fault_re[0][0], val_re[2])
res_fault*100

plot_recon(fault_re[0][1], val_re[2], 'Feedback sensor', False)
res_fault = perf(fault_re[0][1], val_re[2])
res_fault*100

plot_recon(fault_re[0][2], val_re[2], 'Grid anomaly', False)
res_fault = perf(fault_re[0][2], val_re[2])
res_fault*100

plot_recon(fault_re[0][3], val_re[2], 'Array mismatch1', False)
res_fault = perf(fault_re[0][3], val_re[2])
res_fault*100

plot_recon(fault_re[0][4], val_re[2], 'Array mismatch2', False)
res_fault = perf(fault_re[0][4], val_re[2])
res_fault*100


plot_recon(fault_re[0][5], val_re[2], 'Controller', False)
res_fault = perf(fault_re[0][5], val_re[2])
res_fault*100

plot_recon(fault_re[0][6], val_re[2], 'Converter', False)
res_fault = perf(fault_re[0][6], val_re[2])
res_fault*100


np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_0.txt", fault_re[0][0])
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_1.txt", fault_re[0][1])
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_2.txt", fault_re[0][2])
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_3.txt", fault_re[0][3])
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_4.txt", fault_re[0][4])
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_5.txt", fault_re[0][5])
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\ae_error_6.txt", fault_re[0][6])


#Fault data MSE, MAE, MAPE
fault_re[1][0]
fault_re[1][1]
fault_re[1][2]

fault_re[1][3]
fault_re[1][4]
fault_re[1][5]
fault_re[1][6]

results1 = {}

for i in range(7):
   results1[i] = fault_re[1][i]

results1['val'] = val_re[1]
results1['test'] = te_re1[1]

results1_df = pd.DataFrame(results1)
results1_df.T.to_csv(r'C:\a_PhD Research\FDD\Codes\AE\save_dir\ReconPerf.csv')





# =============================================================================
# fault_error_df = []#fault test data
# for i in range(7):
#     _, fed = SPE(fault_re[3][i], fault_re[2][i])
#     fault_error_df.append(fed)
#     
# =============================================================================

#Individual feature's reconstruction error
f=[]    
f0 = error_cont(fault_re[3][0], fault_re[2][0])
f0
f.append(f0)
f1 = error_cont(fault_re[3][1], fault_re[2][1])
f1
f.append(f1)
f2 = error_cont(fault_re[3][2], fault_re[2][2])
f2
f.append(f2)
f3 = error_cont(fault_re[3][3], fault_re[2][3])
f3
f.append(f3)
f4 = error_cont(fault_re[3][4], fault_re[2][4])
f4
f.append(f4)
f5 = error_cont(fault_re[3][5], fault_re[2][5])
f5
f.append(f5)
f6 = error_cont(fault_re[3][6], fault_re[2][6])
f6
f.append(f6)

f_dict ={}
for i in range(7):
    f_dict['Fault_{}'.format(i)] = f[i]

f_dict1 = pd.DataFrame(f_dict)
f_dict1.to_csv(r'C:\a_PhD Research\FDD\Codes\GCN_VAE1\save_dir\Fault_explanantion.csv')
# =============================================================================
# Ipv = ()
# Vpv = ()
# Vdc = ()
# Iabc = ()
# If = ()
# Vabc = ()
# Vf = ()
# for i in range(len(f)):
#     #print(i)
#     for k, v in f[i].items():
#         if k == 'Ipv':
#             Ipv =  Ipv + (f[i][k],)
#         elif k == 'Vpv':
#             Vpv = Vpv + (f[i][k],)
#             
#         elif k == 'Vdc':
#             Vdc = Vdc + (f[i][k],)   
#             
#         elif k == 'Iabc':
#             Iabc = Iabc + (f[i][k],)
#             
#         elif k == 'If':
#             If = If + (f[i][k],)
#             
#         elif k == 'Vabc':
#             Vabc = Vabc + (f[i][k],)
#             
#         elif k == 'Vf':
#             Vf = Vf + (f[i][k],)
#             
#         else:
#             pass
#             
#             #print(f' {k}\t {f[i][k]}')
# 
# features = ('Ipv', 'Vpv', 'Vdc', 'Iabc', 'If', 'Vabc', 'Vf')
# fault_error = {
#     
#     'Inverter': Ipv,
#     'Feedback sensor'
#     
#     } 
# 
# fault_error
# 
# # =============================================================================
# # plot_result2(fault_error_df[6], val_error_df, th)
# # =============================================================================
# 
# # =============================================================================
# # #latent variables
# # 
# # val_hv = latent_vars(model, X_val, edge_indices, edge_weights)
# # te_hv = latent_vars(model, X_te, edge_indices, edge_weights)
# # f1_hv = latent_vars(model, X_fault[6], edge_indices, edge_weights)
# # 
# # 
# # fig, ax = plt.subplots(figsize=(7,5))
# # ax.scatter(val_hv[:, 0], val_hv[:, 1], color='green')
# # ax.scatter(te_hv[:, 0], te_hv[:, 1], cmap='blue')
# # ax.scatter(f1_hv[:, 0], f1_hv[:, 1], cmap='blue')
# # plt.show()
# # 
# # =============================================================================
# =============================================================================


radar_data = pd.read_csv(r"C:\a_PhD Research\FDD\Codes\GCN_VAE1\save_dir\Radra.csv")
radar_data








