# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:28:40 2024

@author: MA11201
"""

import sys
sys.path.append(r'C:\a_PhD Research\FDD\IPPT Exp\GCN_VAE')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("darkgrid")
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['figure.dpi'] = 300
import numpy as np
from GCN_VAE_model import gcnAE
from GCN_VAE_utils import *
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

data_folder = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_35" #folder path of data
model_path = r"C:\a_PhD Research\FDD\IPPT Exp\save_model_cosine7000\GCN_VAE_wind35.pkl" # pretrained model path

#loading the pretrained model

model = gcnAE(13)
model.load_state_dict(torch.load(model_path))
#creating normal datasets
X_tr = data_read(data_folder, 'train', 0, 7000)
X_val = data_read(data_folder, 'val', 7000, 9000)
X_test = data_read(data_folder, 'test', 9000, 10000)
X_tr.shape, X_val.shape, X_test.shape
#creating fault datasets
X_fault = data_read(data_folder, 'fault', 0, 1000)
#creating graph attributes #spatial-metric=Cosine
edge_indices, edge_weights =  graph_attr(X_tr, 'Cosine')
X_tr = torch.tensor(np.array(X_tr), dtype=torch.float)
X_val = torch.tensor(np.array(X_val), dtype=torch.float)
X_test = torch.tensor(np.array(X_test), dtype=torch.float)


#evaluation
model = model.eval()
conf_value = 0.99

#validation
val_re = eval_normal(model, X_val, conf_value, edge_indices, edge_weights, 'val')
plot_recon(val_re[0], val_re[2], 'Validation', True)
val_re[1]

#Histogram plot of reconstructuion error and PDF
# =============================================================================
# fig, ax = plt.subplots(figsize=(5, 3))
# sns.kdeplot(np.array(val_re[0]), fill = True, ax=ax)
# ax.axvline(val_re[2], color = 'red', ls = '-.', label='Threshold')#
# ax.set_xlabel('Overall reconstruction error')
# ax.legend(loc= 'upper right')
# plt.savefig(r"C:\a_PhD Research\FDD\IPPT Exp\Figures\KDE&Th.pdf", bbox_inches='tight')
# plt.tight_layout()
# plt.show()
# =============================================================================


#Normal test
te_re1 = eval_normal(model, X_test, conf_value, edge_indices, edge_weights, 'test')
plot_recon(te_re1[0], val_re[2], 'Normal_test', True)
te_re1[1]
#False alarm rate
res_val = perf(val_re[0], val_re[2])
res_test = perf(te_re1[0], val_re[2])
res_val*100, res_test*100

fault_re = eval_fault(model, X_fault, conf_value, edge_indices, edge_weights, 'fault')
for i in range(7):   
    fault_res = perf(fault_re[0][i], val_re[2])
    print(fault_res*100)




#Fault data
plot_recon(fault_re[0][0], val_re[2], 'Inverter', False)
plot_recon(fault_re[0][1], val_re[2], 'Feedback sensor', False)
plot_recon(fault_re[0][2], val_re[2], 'Grid anomaly', False)
plot_recon(fault_re[0][3], val_re[2], 'Array mismatch1', False)
plot_recon(fault_re[0][4], val_re[2], 'Array mismatch2', False)
plot_recon(fault_re[0][5], val_re[2], 'Controller', False)
plot_recon(fault_re[0][6], val_re[2], 'Converter', False)


#latent_variable plotting
# =============================================================================
# dv = latent_vars(model, X_val, edge_indices, edge_weights)
# dt = latent_vars(model, X_test, edge_indices, edge_weights)
# 
# df1 = latent_vars(model, X_fault[0], edge_indices, edge_weights)
# df2 = latent_vars(model, X_fault[1], edge_indices, edge_weights)
# df3 = latent_vars(model, X_fault[2], edge_indices, edge_weights)
# df4 = latent_vars(model, X_fault[3], edge_indices, edge_weights)
# df5 = latent_vars(model, X_fault[4], edge_indices, edge_weights)
# df6 = latent_vars(model, X_fault[5], edge_indices, edge_weights)
# df7 = latent_vars(model, X_fault[6], edge_indices, edge_weights)
# 
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# #ax = fig.add_subplot()
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.scatter(dv[:, 0], dv[:, 1])
# ax.scatter(dt[:, 0], dt[:, 1])
# ax.scatter(df1[:, 0], df1[:, 1])
# # =============================================================================
# # ax.scatter(df2[:, 0], df2[:, 1])
# # ax.scatter(df3[:, 0], df3[:, 1])
# # ax.scatter(df4[:, 0], df4[:, 1])
# # ax.scatter(df5[:, 0], df5[:, 1])
# # ax.scatter(df6[:, 0], df6[:, 1])
# # ax.scatter(df7[:, 0], df7[:, 1])
# # =============================================================================
# plt.show()
# 
# 
# pca = PCA(n_components=1)
# dv_pca = pca.fit_transform(dv)
# dv_pca.shape
# dt_pca = pca.fit_transform(dt)
# dt_pca.shape
# df1_pca = pca.fit_transform(df6)
# df1_pca.shape
# 
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(np.arange(dv_pca.shape[0]), dv_pca)
# ax.scatter(np.arange(dt_pca.shape[0]), dt_pca)
# ax.scatter(np.arange(df1_pca.shape[0]), df1_pca)
# plt.show()
# 
# =============================================================================




# =============================================================================
# for i in range(7):
#     np.savetxt(r"C:\a_PhD Research\FDD\Codes\GCN_VAE1\save_dir\fault_error{}.txt".format(i),fault_re[0][i])
# 
# =============================================================================

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
results1_df.T.to_csv(r'C:\a_PhD Research\FDD\IPPT Exp\Results\ReconPerf_wind5.csv')

#Model validation by intentional feature modification (Fault localization)

tr_data = data_read(data_folder, 'train', 0, 7000)
tr_data.shape

tr_data1 = torch.tensor(np.array(tr_data), dtype=torch.float)

tr_pred = model(tr_data1, edge_indices, edge_weights)
tr_pred1 = tr_pred[0].detach().numpy()
tr_pred2 = pd.DataFrame(tr_pred1, columns = tr_data.columns)


#reconstruction plot
# =============================================================================
# fig, axs= plt.subplots(nrows=7, ncols=2, figsize=(11,10)) #(width, height)
# plt.subplots_adjust(hspace=0.3)
# plt.subplots_adjust(wspace =0.3)
# cols = tr_data.columns
# for c, ax in zip(cols, axs.ravel()):
#   ax.plot(tr_data[c], label='o')
#   ax.plot(tr_pred2[c], label='p')
#   ax.legend()
#   ax.set_xlabel('data points')
#   ax.set_ylabel(c)
# #plt.savefig(r'C:\a_PhD Research\FDD\Codes\GCN_VAE1\save_dir\Figures\Reconstruction.pdf')
# plt.tight_layout()
# plt.show()
# =============================================================================

#error contribution of each feature
tr_data
ec=error_cont(np.array(tr_pred2), np.array(tr_data))
ec1 = pd.DataFrame(ec, index=[0])
ec1.to_csv(r'C:\a_PhD Research\FDD\IPPT Exp\Results\BeforeNoise_FeatureMSE_wind5.csv')




noise = np.random.normal(1.5, 3.5, tr_data.shape[0])
noise.shape
cols = tr_data.columns

# =============================================================================
# noise1 = np.random.normal(1.5, 3.5, tr_data.shape[0])
# #Intentional noise adding plot
# noisy_feature1 = tr_data['Ipv'].iloc[:700] + noise[:700]
# noisy_feature2 = tr_data['Ipv'].iloc[:700] + noise1[:700]
# 
# fig, ax =plt.subplots(figsize=(5, 3))
# ax.plot(noisy_feature1, label = 'Noise1')
# ax.plot(noisy_feature2, label = 'Noise2')
# ax.plot(tr_data['Ipv'].iloc[:700], label = 'Original')
# ax.set_xlabel('Data points')
# ax.set_ylabel('Ipv')
# ax.grid(visible=True, which='major', axis='both')
# ax.legend(loc='upper right')
# plt.savefig(r"C:\a_PhD Research\FDD\IPPT Exp\Figures\Artificial_Noise.pdf", bbox_inches='tight')
# plt.tight_layout()
# plt.show()
# 
# =============================================================================



error_test = {}
for i in range(13):    
    tr_fea = tr_data[cols[i]]
    fea_noise = tr_fea + noise
    tr_data_noise = tr_data.copy()    
    tr_data_noise[cols[i]] = fea_noise
    tr_data_noise1 = torch.tensor(np.array(tr_data_noise), dtype=torch.float)
    tr_data_noise_pred = model(tr_data_noise1, edge_indices, edge_weights)
    tr_data_noise_pred1 = tr_data_noise_pred[0].detach().numpy()
    error_test['key_{}'.format(i)] = error_cont(np.array(tr_data_noise_pred1), np.array(tr_data_noise))


error_test
error_test1 = pd.DataFrame(error_test)
error_test1.to_csv(r'C:\a_PhD Research\FDD\IPPT Exp\Results\error_test_wind5.csv')


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
f_dict1.to_csv(r'C:\a_PhD Research\FDD\IPPT Exp\Results\Fault_explanantion_wind5.csv')


#==============================================================================
#Individual feature reconstruction for normal test data
test_feature_mse = error_cont(te_re1[2], te_re1[3])
test_feature_mse

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








